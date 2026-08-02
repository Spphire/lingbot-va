# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import random
import sys
import time
from pathlib import Path
import wandb

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.deepspeed import (
    materialize_deepspeed_config,
    prepare_accumulated_losses,
    validate_distributed_backend,
)
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
from wan_va.dataset.nmx_action_adapter import (
    rebuild_batch_actions,
    uses_nmx_action_contract,
)
from wan_va.dataset.empty_embedding import generate_empty_embedding
from wan_va.run_contract import persist_run_contracts, write_checkpoint_manifest
import gc


def apply_action_history_condition_dropout(action_dict, probability):
    """Zero normalized clean action-history samples without changing targets."""
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "action_history_condition_dropout_prob must be within [0, 1], "
            f"got {probability}"
        )

    clean_action = action_dict["latent"]
    batch_size = int(clean_action.shape[0])
    if probability == 0.0:
        return torch.zeros((), dtype=torch.int64, device=clean_action.device)
    if probability == 1.0:
        clean_action.zero_()
        return torch.full(
            (), batch_size, dtype=torch.int64, device=clean_action.device
        )

    drop_mask = torch.rand(batch_size, device=clean_action.device) < probability
    clean_action.masked_fill_(
        drop_mask.view(batch_size, *([1] * (clean_action.ndim - 1))),
        0,
    )
    return drop_mask.sum()


def append_metrics_jsonl(path, row):
    """Durably append one optimizer-step metric row for MLflow sidecars."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def seed_training_process(seed, rank):
    process_seed = int(seed) + int(rank)
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    torch.cuda.manual_seed_all(process_seed)


class Trainer:
    def __init__(self, config, run_contract=None):
        if config.enable_wandb and config.rank == 0:
            wandb.login(host=os.environ['WANDB_BASE_URL'], key=os.environ['WANDB_API_KEY'])
            self.wandb = wandb
            self.wandb.init(
                entity=os.environ["WANDB_TEAM_NAME"],
                project=os.getenv("WANDB_PROJECT", "va_robotwin"),
                # dir=log_dir,
                config=config,
                mode="online",
                name='test_lln'
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.run_contract = run_contract
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.distributed_backend = validate_distributed_backend(
            config.distributed_backend
        )
        self.deepspeed_engine = None
        self.gradient_accumulation_steps = int(
            getattr(config, 'gradient_accumulation_steps', 1)
        )
        self.action_history_dropout_prob = float(
            getattr(config, "action_history_condition_dropout_prob", 0.0)
        )
        action_history_dropout_mode = getattr(
            config,
            "action_history_condition_dropout_mode",
            "zero_normalized_clean_condition",
        )
        if (
            self.action_history_dropout_prob > 0.0
            and action_history_dropout_mode != "zero_normalized_clean_condition"
        ):
            raise ValueError(
                "Unsupported action-history dropout mode: "
                f"{action_history_dropout_mode!r}"
            )
        if not 0.0 <= self.action_history_dropout_prob <= 1.0:
            raise ValueError(
                "action_history_condition_dropout_prob must be within [0, 1], "
                f"got {self.action_history_dropout_prob}"
            )
        self._latest_action_history_dropped_samples = torch.zeros(
            (), dtype=torch.int64, device=self.device
        )

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            attn_mode="flex"
        )

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        if self._uses_deepspeed:
            logger.info("Setting up DeepSpeed model...")
            self.transformer.to(device=self.device, dtype=self.dtype)
        else:
            logger.info("Setting up FSDP...")
            self.transformer = _configure_model(
                model=self.transformer,
                shard_fn=shard_model,
                param_dtype=self.dtype,
                device=self.device,
                eval_mode=False,
            )
        self.transformer.train()
        self.transformer.requires_grad_(True)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            [p for p in self.transformer.parameters() if p.requires_grad],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        if self._uses_deepspeed:
            self._initialize_deepspeed_engine()

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(
            config=config,
            num_init_worker=config.num_init_worker,
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=int(getattr(config, "seed", 42)),
        ) if config.world_size > 1 else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None), 
            num_workers=config.load_worker,
            sampler=train_sampler,
        )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.run_dir = Path(config.save_root)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.save_dir = self.run_dir / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.train_loader_iter = None
        if hasattr(config, 'resume_from') and config.resume_from:
            self._load_training_state(config.resume_from)

    @property
    def _uses_deepspeed(self):
        return self.distributed_backend == 'deepspeed'

    def _initialize_deepspeed_engine(self):
        try:
            import deepspeed
        except ImportError as exc:
            raise ImportError(
                "DeepSpeed backend requested; install with `pip install .[deepspeed]`"
            ) from exc

        ds_config = materialize_deepspeed_config(
            self.config.deepspeed_config_file,
            micro_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            world_size=self.config.world_size,
            param_dtype=self.dtype,
            gradient_clipping=self.config.gradient_clipping,
        )
        engine, optimizer, _, scheduler = deepspeed.initialize(
            model=self.transformer,
            model_parameters=[
                p for p in self.transformer.parameters() if p.requires_grad
            ],
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            config=ds_config,
        )
        self.deepspeed_engine = engine
        self.transformer = engine
        self.optimizer = optimizer
        if scheduler is not None:
            self.lr_scheduler = scheduler
        if self.config.rank == 0:
            logger.info(
                "DeepSpeed initialized: ZeRO stage %s, global batch size %s",
                ds_config.get('zero_optimization', {}).get('stage'),
                ds_config.get('train_batch_size'),
            )

    def _unwrap_transformer(self):
        if self.deepspeed_engine is not None:
            return self.deepspeed_engine.module
        return self.transformer
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        nmx_action_contract = uses_nmx_action_contract(self.config)
        if nmx_action_contract:
            chunk_size = torch.randint(
                int(getattr(self.config, 'action_chunk_size_min', 1)),
                int(getattr(self.config, 'action_chunk_size_max', 4)) + 1,
                (1,),
            ).item()
            window_size = torch.randint(
                int(getattr(self.config, 'window_size_min', 4)),
                int(getattr(self.config, 'window_size_max', 64)) + 1,
                (1,),
            ).item()
            batch_actions, batch_action_masks = rebuild_batch_actions(
                batch_dict,
                self.config,
                chunk_size,
            )
        else:
            batch_actions = batch_dict['actions']
            batch_action_masks = batch_dict['actions_mask']

        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_actions,
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_action_masks,
            action_mode=True,
            noisy_cond_prob=0.0)

        self._latest_action_history_dropped_samples = (
            apply_action_history_condition_dropout(
                action_dict,
                self.action_history_dropout_prob,
            )
        )

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        action_dict['actions_mask'] = batch_action_masks

        if not nmx_action_contract:
            # Keep the upstream RNG order unchanged for existing training configs.
            chunk_size = torch.randint(1, 5, (1,)).item()
            window_size = torch.randint(4, 65, (1,)).item()

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': chunk_size,
            'window_size': window_size,
            'chunk_grouping_start_from_one': bool(
                getattr(self.config, 'chunk_grouping_start_from_one', False)
            ),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss, action_loss

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        if not self._uses_deepspeed:
            self.transformer.set_requires_gradient_sync(should_sync)

        output = self.transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss_payload = prepare_accumulated_losses(
            latent_loss,
            action_loss,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            uses_deepspeed=self._uses_deepspeed,
        )

        losses = {
            'latent_loss': loss_payload['latent_loss_for_log'].detach(),
            'action_loss': loss_payload['action_loss_for_log'].detach(),
            'action_history_dropped_samples': (
                self._latest_action_history_dropped_samples.detach()
            ),
        }

        if self._uses_deepspeed:
            self.deepspeed_engine.backward(loss_payload['backward_loss'])
            global_steps_before = int(self.deepspeed_engine.global_steps)
            self.deepspeed_engine.step()
            did_step = int(self.deepspeed_engine.global_steps) > global_steps_before
            losses['should_log'] = did_step
            if did_step:
                total_norm = self.deepspeed_engine.get_global_grad_norm()
                losses['total_norm'] = torch.as_tensor(
                    0.0 if total_norm is None else total_norm,
                    device=self.device,
                )
        else:
            loss_payload['backward_loss'].backward()
            if should_sync:
                total_norm = torch.nn.utils.clip_grad_norm_(
                    self.transformer.parameters(),
                    self.config.gradient_clipping,
                )
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                losses['total_norm'] = total_norm
                losses['should_log'] = True
            else:
                losses['should_log'] = False

        return losses

    def _save_transformer_artifacts(self, checkpoint_dir, state_dict, config):
        transformer_dir = checkpoint_dir / "transformer"
        transformer_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving transformer to {transformer_dir}")
        save_file(
            {k: v.detach().cpu().to(torch.bfloat16) for k, v in state_dict.items()},
            transformer_dir / "diffusion_pytorch_model.safetensors",
        )
        config_dict = dict(config)
        config_dict.pop('_name_or_path', None)
        with (transformer_dir / "config.json").open('w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)

    def save_checkpoint(self):
        """Save model weights and resumable backend state."""
        checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
        if self._uses_deepspeed:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self.deepspeed_engine.save_checkpoint(
                str(checkpoint_dir / 'deepspeed'),
                tag=f"global_step{self.step}",
                client_state={'step': self.step},
            )
            if self.config.rank == 0:
                module = self._unwrap_transformer()
                self._save_transformer_artifacts(
                    checkpoint_dir,
                    module.state_dict(),
                    module.config,
                )
                if self.run_contract is not None:
                    write_checkpoint_manifest(
                        checkpoint_dir,
                        step=self.step,
                        run_contract=self.run_contract,
                    )
        else:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            if self.config.rank == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                self._save_transformer_artifacts(
                    checkpoint_dir,
                    state_dict,
                    self._unwrap_transformer().config,
                )
                if self.run_contract is not None:
                    write_checkpoint_manifest(
                        checkpoint_dir,
                        step=self.step,
                        run_contract=self.run_contract,
                    )

        if dist.is_initialized():
            dist.barrier()
        if self.config.rank == 0:
            logger.info(f"Checkpoint saved successfully at step {self.step}")

    def _load_training_state(self, checkpoint_path):
        """Load backend optimizer/scheduler state after model initialization."""
        checkpoint_dir = Path(checkpoint_path)
        if self._uses_deepspeed:
            deepspeed_dir = checkpoint_dir / 'deepspeed'
            load_path, client_state = self.deepspeed_engine.load_checkpoint(
                str(deepspeed_dir)
            )
            if load_path is None:
                raise FileNotFoundError(
                    f"DeepSpeed training state not found: {deepspeed_dir}"
                )
            self.step = int((client_state or {}).get('step', 0))
            if self.config.rank == 0:
                logger.info(
                    f"DeepSpeed state loaded from {load_path}, resuming at step {self.step}"
                )
            return

        training_state_path = checkpoint_dir / "training_state.pt"

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)
        set_optimizer_state_dict(
            self.transformer, self.optimizer,
            optim_state_dict=training_state['optimizer_state_dict'],
            options=StateDictOptions(full_state_dict=True, strict=False)
        )
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        if dist.is_initialized():
            dist.barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_action_history_dropped_samples = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_action_history_dropped_samples.append(
                losses['action_history_dropped_samples']
            )
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                action_history_dropped_show = (
                    dist_mean(
                        torch.stack(accumulated_action_history_dropped_samples)
                        .sum()
                        .float()
                    )
                    * int(self.config.world_size)
                ).detach().cpu().item()

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_action_history_dropped_samples = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    max_latent_frames = getattr(
                        self.config,
                        'max_latent_frames',
                        0,
                    )
                    progress_bar.n += 1
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}'
                    })
                    metric_row = {
                        'step': self.step,
                        'loss_metrics/global_avg_video_loss': latent_loss_show,
                        'loss_metrics/global_avg_action_loss': action_loss_show,
                        'loss_metrics/global_max_video_loss': max_latent_loss_show,
                        'loss_metrics/global_max_action_loss': max_action_loss_show,
                        'grad_norm': total_norm.item(),
                        'lr': lr,
                        'train/sequence_parallel_size': 1,
                        'train/data_parallel_world_size': int(self.config.world_size),
                        'train/physical_world_size': int(self.config.world_size),
                        'train/global_batch_size': (
                            int(self.config.batch_size)
                            * self.gradient_accumulation_steps
                            * int(self.config.world_size)
                        ),
                        'train/action_history_condition_dropout_enabled': int(
                            self.action_history_dropout_prob > 0.0
                        ),
                        'train/action_history_condition_dropout_prob': (
                            self.action_history_dropout_prob
                        ),
                        'train/action_history_condition_dropout_dropped_samples': int(
                            round(action_history_dropped_show)
                        ),
                        'data/allowed_max_latent_frames': (
                            None
                            if max_latent_frames is None
                            else int(max_latent_frames)
                        ),
                        'timestamp': time.time(),
                    }
                    append_metrics_jsonl(self.metrics_path, metric_row)
                    if self.config.enable_wandb:
                        self.wandb.log({
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.seed = int(getattr(config, "seed", 42))
    seed_training_process(config.seed, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    config.distributed_backend = validate_distributed_backend(
        args.distributed_backend
    )
    config.deepspeed_config_file = args.deepspeed_config
    config.resume_from = args.resume_from
    config.num_init_worker = (
        args.num_init_workers
        if args.num_init_workers is not None
        else int(getattr(config, 'num_init_worker', 8))
    )
    config.gradient_clipping = float(
        getattr(config, 'gradient_clipping', 2.0)
    )

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
        config.empty_emb_path = os.path.join(args.dataset_path, 'empty_emb.pt')
    if args.empty_emb_path is not None:
        config.empty_emb_path = args.empty_emb_path
    if args.model_path is not None:
        config.wan22_pretrained_model_name_or_path = args.model_path
    if args.num_workers is not None:
        config.load_worker = args.num_workers
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.save_interval is not None:
        config.save_interval = args.save_interval
    if args.disable_wandb:
        config.enable_wandb = False
    if args.dataset_manifest is not None:
        config.dataset_manifest_path = args.dataset_manifest

    if float(getattr(config, 'cfg_prob', 0.0)) > 0:
        configured_empty_emb = Path(config.empty_emb_path)
        if configured_empty_emb.is_file():
            empty_emb_path = configured_empty_emb
        elif args.empty_emb_path is not None:
            empty_emb_path = configured_empty_emb
        else:
            empty_emb_path = Path(config.save_root) / "cache" / "empty_emb.pt"
        config.empty_emb_path = str(empty_emb_path)

        if rank == 0 and not empty_emb_path.is_file():
            logger.info(
                "Generating empty prompt embedding at %s",
                empty_emb_path,
            )
            generate_empty_embedding(
                config.wan22_pretrained_model_name_or_path,
                empty_emb_path,
                device=torch.device(f"cuda:{local_rank}"),
            )
        if world_size > 1:
            dist.barrier(device_ids=[local_rank])
        if not empty_emb_path.is_file():
            raise FileNotFoundError(
                f"Empty prompt embedding was not created: {empty_emb_path}"
            )

    run_contract = None
    if getattr(config, "action_contract", None) == "nmx_chunk_relative_v10":
        dataset_manifest_path = getattr(config, "dataset_manifest_path", None)
        if not dataset_manifest_path:
            raise ValueError(
                "NMX training requires --dataset-manifest so the run is self-describing"
            )
        if rank == 0:
            run_contract = persist_run_contracts(
                config,
                config_name=args.config_name,
                launch_args=vars(args),
                dataset_manifest_path=dataset_manifest_path,
                repo_root=Path(__file__).resolve().parents[1],
            )

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(
            f"Backend: {config.distributed_backend}, world size: {world_size}, "
            f"local rank: {local_rank}"
        )

    trainer = Trainer(config, run_contract=run_contract)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--distributed-backend",
        choices=sorted(('fsdp', 'deepspeed')),
        default='fsdp',
    )
    parser.add_argument(
        "--deepspeed-config",
        default="config/deepspeed/zero2.json",
    )
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-manifest", default=None)
    parser.add_argument("--empty-emb-path", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--num-init-workers", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--local_rank", type=int, default=-1, help=argparse.SUPPRESS)

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    init_logger()
    main()
