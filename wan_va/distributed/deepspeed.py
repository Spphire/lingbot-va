import json
from copy import deepcopy
from pathlib import Path

import torch


SUPPORTED_DISTRIBUTED_BACKENDS = {"fsdp", "deepspeed"}


def validate_distributed_backend(backend: str) -> str:
    backend = str(backend).strip().lower()
    if backend not in SUPPORTED_DISTRIBUTED_BACKENDS:
        supported = ", ".join(sorted(SUPPORTED_DISTRIBUTED_BACKENDS))
        raise ValueError(
            f"Unsupported distributed backend {backend!r}; expected one of: {supported}"
        )
    return backend


def materialize_deepspeed_config(
    config_path,
    *,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    param_dtype,
    gradient_clipping: float,
):
    path = Path(config_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as f:
        config = deepcopy(json.load(f))

    values = {
        "train_micro_batch_size_per_gpu": int(micro_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "train_batch_size": (
            int(micro_batch_size)
            * int(gradient_accumulation_steps)
            * int(world_size)
        ),
    }
    for key, value in values.items():
        if config.get(key) == "auto":
            config[key] = value
        elif int(config.get(key)) != value:
            raise ValueError(
                f"DeepSpeed {key}={config.get(key)} does not match runtime value {value}"
            )

    if config.get("gradient_clipping") == "auto":
        config["gradient_clipping"] = float(gradient_clipping)

    if isinstance(config.get("bf16"), dict):
        if config["bf16"].get("enabled") == "auto":
            config["bf16"]["enabled"] = param_dtype == torch.bfloat16
    if isinstance(config.get("fp16"), dict):
        if config["fp16"].get("enabled") == "auto":
            config["fp16"]["enabled"] = param_dtype == torch.float16

    return config


def prepare_accumulated_losses(
    latent_loss,
    action_loss,
    *,
    gradient_accumulation_steps: int,
    uses_deepspeed: bool,
):
    accumulation_steps = int(gradient_accumulation_steps)
    if accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")

    backward_loss = latent_loss + action_loss
    if not uses_deepspeed:
        backward_loss = backward_loss / accumulation_steps

    return {
        "backward_loss": backward_loss,
        "latent_loss_for_log": latent_loss / accumulation_steps,
        "action_loss_for_log": action_loss / accumulation_steps,
    }
