# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import get_episode_data_index
from lerobot.datasets.compute_stats import aggregate_stats, compute_episode_stats
import hashlib
import numpy as np
from pathlib import Path
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
import os
import threading
from tqdm import tqdm
from functools import partial
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME

from wan_va.dataset.nmx_action_adapter import (
    apply_episode_action_validity,
    build_model_actions_from_raw,
    infer_sampled_video_frames_per_latent,
    load_action_norm_stats,
    prepare_raw_action_tensor,
    uses_nmx_action_contract,
)


UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT = "upstream_single_canvas_v1"
PER_VIEW_ZERO_PAD_VISUAL_CONTRACT = "per_view_zero_pad_then_concat_v1"
_NMX_VISUAL_CONTRACTS = {
    UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT,
    PER_VIEW_ZERO_PAD_VISUAL_CONTRACT,
}
_TEXT_EMB_OVERRIDE_CACHE = {}
_TEXT_EMB_OVERRIDE_CACHE_LOCK = threading.Lock()


def _load_text_emb_override(config):
    override_path = getattr(config, "text_emb_override_path", None)
    if not override_path:
        return None

    path = Path(override_path).expanduser().resolve()
    with _TEXT_EMB_OVERRIDE_CACHE_LOCK:
        cached = _TEXT_EMB_OVERRIDE_CACHE.get(path)
        if cached is None:
            if not path.is_file():
                raise FileNotFoundError(f"Text embedding override not found: {path}")
            tensor = torch.load(path, map_location="cpu", weights_only=False)
            if not torch.is_tensor(tensor):
                raise TypeError(
                    f"Text embedding override must contain a tensor, got {type(tensor)!r}"
                )
            tensor = tensor.detach().cpu().contiguous()
            digest = hashlib.sha256(
                tensor.view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            cached = (tensor, digest)
            _TEXT_EMB_OVERRIDE_CACHE[path] = cached

    tensor, digest = cached
    expected_shape = getattr(config, "text_emb_override_shape", None)
    if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
        raise ValueError(
            f"Text embedding override shape mismatch for {path}: "
            f"expected {tuple(expected_shape)}, got {tuple(tensor.shape)}"
        )
    expected_dtype = getattr(config, "text_emb_override_dtype", None)
    if expected_dtype is not None and str(tensor.dtype) != str(expected_dtype):
        raise ValueError(
            f"Text embedding override dtype mismatch for {path}: "
            f"expected {expected_dtype}, got {tensor.dtype}"
        )
    expected_digest = getattr(config, "text_emb_override_sha256", None)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError(
            f"Text embedding override SHA-256 mismatch for {path}: "
            f"expected {expected_digest}, got {digest}"
        )
    return tensor


def _pad_latent_view_to_patch_size(latent, patch_size):
    """Right/bottom-pad one FHWC view without changing its temporal contract."""
    patch_f, patch_h, patch_w = (int(value) for value in patch_size)
    if min(patch_f, patch_h, patch_w) <= 0:
        raise ValueError(f"patch_size must be positive, got {tuple(patch_size)}")
    if patch_f != 1:
        raise ValueError(
            "The visual-only padding control requires temporal patch size 1, "
            f"got {patch_f}"
        )

    _, height, width, _ = latent.shape
    pad_height = (-height) % patch_h
    pad_width = (-width) % patch_w
    if not (pad_height or pad_width):
        return latent

    latent = latent.permute(3, 0, 1, 2)
    latent = F.pad(
        latent,
        (0, pad_width, 0, pad_height),
        mode="constant",
        value=0,
    )
    return latent.permute(1, 2, 3, 0).contiguous()


def compose_nmx_latent_views(latent_views, config):
    """Compose NMX wrist views while keeping the selected visual contract explicit."""
    if not latent_views:
        raise ValueError("NMX visual contract requires at least one latent view")
    if any(latent.ndim != 4 for latent in latent_views):
        shapes = [tuple(latent.shape) for latent in latent_views]
        raise ValueError(f"NMX latent views must be FHWC tensors, got {shapes}")

    visual_contract = getattr(
        config,
        "visual_contract",
        UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT,
    )
    if visual_contract not in _NMX_VISUAL_CONTRACTS:
        raise ValueError(
            f"Unsupported NMX visual_contract {visual_contract!r}; "
            f"expected one of {sorted(_NMX_VISUAL_CONTRACTS)}"
        )

    actual_view_shapes = [
        (int(latent.shape[1]), int(latent.shape[2])) for latent in latent_views
    ]
    expected_view_shapes = getattr(config, "expected_latent_view_shapes", None)
    if expected_view_shapes is not None:
        expected_view_shapes = [tuple(map(int, shape)) for shape in expected_view_shapes]
        if actual_view_shapes != expected_view_shapes:
            raise ValueError(
                "NMX latent view geometry does not match the configured visual "
                f"contract: expected {expected_view_shapes}, got {actual_view_shapes}"
            )

    frame_counts = {int(latent.shape[0]) for latent in latent_views}
    channel_counts = {int(latent.shape[3]) for latent in latent_views}
    if len(frame_counts) != 1 or len(channel_counts) != 1:
        raise ValueError(
            "NMX latent views must share frame and channel counts, got "
            f"{[tuple(latent.shape) for latent in latent_views]}"
        )
    expected_channels = getattr(config, "expected_latent_channels", None)
    if expected_channels is not None and channel_counts != {int(expected_channels)}:
        raise ValueError(
            f"NMX latent views must have {int(expected_channels)} channels, "
            f"got {sorted(channel_counts)}"
        )

    patch_size = tuple(int(value) for value in config.patch_size)
    if visual_contract == PER_VIEW_ZERO_PAD_VISUAL_CONTRACT:
        latent_views = [
            _pad_latent_view_to_patch_size(latent, patch_size)
            for latent in latent_views
        ]

    try:
        composed = torch.cat(latent_views, dim=2)
    except RuntimeError as exc:
        raise ValueError(
            "NMX single-canvas views must have compatible frame and height dimensions"
        ) from exc

    composed_fhw = tuple(int(value) for value in composed.shape[:3])
    if any(size % patch != 0 for size, patch in zip(composed_fhw, patch_size)):
        raise ValueError(
            f"Composed latent shape {composed_fhw} is not divisible by patch_size "
            f"{patch_size} under visual_contract={visual_contract!r}"
        )
    return composed


def ensure_hf_datasets_list_compat():
    """Teach datasets<=3.6 to read parquet metadata written by datasets 4."""
    from datasets.features import features as feature_module

    if "List" not in feature_module._FEATURE_TYPES:
        feature_module._FEATURE_TYPES["List"] = feature_module.Sequence


def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def discover_dataset_roots(dataset_path):
    paths = (
        list(dataset_path)
        if isinstance(dataset_path, (list, tuple))
        else [dataset_path]
    )
    roots = []
    for configured_path in paths:
        configured_path = os.fspath(configured_path)
        if os.path.isfile(os.path.join(configured_path, 'meta', 'info.json')):
            roots.append(configured_path)
            continue
        roots.extend(
            value.split('/meta/info.json')[0]
            for value in recursive_find_file(configured_path, 'info.json')
        )
    # Preserve the training-record order while avoiding duplicate roots.
    return list(dict.fromkeys(roots))

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    if uses_nmx_action_contract(config):
        repo_list = discover_dataset_roots(config.dataset_path)
    else:
        repo_list = recursive_find_file(config.dataset_path, 'info.json')
        repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    if len(repo_list) <= 1 or num_init_worker <= 1:
        return [construct_func(repo_id) for repo_id in repo_list]

    # Dataset setup happens after CUDA and the process group are initialized.
    # Threads avoid forking the fully loaded model into every init worker.
    max_workers = min(num_init_worker, len(repo_list))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(construct_func, repo_list))

def get_relative_pose(pose):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
    ):
        ensure_hf_datasets_list_compat()
        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=False
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.nmx_action_contract = uses_nmx_action_contract(config)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.latent_path = Path(repo_id) / 'latents'
        self.empty_emb = None
        if self.cfg_prob > 0:
            self.empty_emb = torch.load(
                config.empty_emb_path,
                weights_only=False,
            )
        self.text_emb_override = _load_text_emb_override(config)
        self.used_video_keys = config.obs_cam_keys
        if self.nmx_action_contract:
            norm_stat = load_action_norm_stats(self.root, config)
        else:
            norm_stat = config.norm_stat
        norm_dtype = 'float32' if self.nmx_action_contract else 'float'
        self.q01 = np.array(norm_stat['q01'], dtype=norm_dtype)[None]
        self.q99 = np.array(norm_stat['q99'], dtype=norm_dtype)[None]
        columns = ['action']
        self.state_column = None
        if self.nmx_action_contract:
            self.state_column = getattr(config, 'state_column', 'observation.state')
            if self.state_column not in self.hf_dataset.column_names:
                raise ValueError(
                    f"{self.root}: NMX action contract requires parquet column "
                    f"{self.state_column!r}"
                )
            columns.append(self.state_column)
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=columns,
                output_all_columns=False
            )
        self.parse_meta()

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                if self.nmx_action_contract:
                    cur_meta["slam_validity_all"] = bool(
                        value.get("slam_validity_all", True)
                    )
                    cur_meta["width_validity"] = bool(
                        value.get("width_detection", True)
                        and value.get("width_calibration", True)
                    )
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
    def _clip_temporal_latents(self, data_dict):
        max_latent_frames = getattr(self.config, 'max_latent_frames', None)
        if not self.nmx_action_contract or max_latent_frames is None:
            return data_dict

        reference_key = self.used_video_keys[0]
        source_latent_frames = int(data_dict[f"{reference_key}.latent_num_frames"])
        clip_frames = min(source_latent_frames, int(max_latent_frames))
        if clip_frames == source_latent_frames:
            return data_dict
        clip_start = int(np.random.randint(0, source_latent_frames - clip_frames + 1))
        clip_end = clip_start + clip_frames

        for key in self.used_video_keys:
            key_latent_frames = int(data_dict[f"{key}.latent_num_frames"])
            if key_latent_frames != source_latent_frames:
                raise ValueError(
                    f"Latent frame mismatch across views: {key} has {key_latent_frames}, "
                    f"expected {source_latent_frames}"
                )
            latent_height = int(data_dict[f"{key}.latent_height"])
            latent_width = int(data_dict[f"{key}.latent_width"])
            tokens_per_frame = latent_height * latent_width
            data_dict[f"{key}.latent"] = data_dict[f"{key}.latent"][
                clip_start * tokens_per_frame : clip_end * tokens_per_frame
            ].contiguous()
            data_dict[f"{key}.latent_num_frames"] = clip_frames

            frame_ids = list(data_dict[f"{key}.frame_ids"])
            source_video_frames = int(
                data_dict.get(f"{key}.video_num_frames", len(frame_ids))
            )
            sampled_per_latent = infer_sampled_video_frames_per_latent(
                source_video_frames, source_latent_frames
            )
            frame_start = min(clip_start * sampled_per_latent, len(frame_ids) - 1)
            frame_count = (clip_frames - 1) * sampled_per_latent + 1
            clipped_ids = frame_ids[frame_start : frame_start + frame_count]
            data_dict[f"{key}.frame_ids"] = clipped_ids
            data_dict[f"{key}.video_num_frames"] = len(clipped_ids)
        return data_dict

        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                 f=latent_num_frames, 
                                 h=latent_height, 
                                 w=latent_width)
            latent_lst.append(latent)
        if self.config.env_type == 'robotwin_tshape':
            wrist_latent = torch.cat(latent_lst[1:], dim=2)
            cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)
        elif self.nmx_action_contract:
            cat_latent = compose_nmx_latent_views(latent_lst, self.config)
        else:
            cat_latent = torch.cat(latent_lst, dim=2)

        text_emb = self.text_emb_override
        if text_emb is None:
            text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if self.empty_emb is not None and torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process(
        self,
        local_start_frame,
        local_end_frame,
        latent_frame_ids,
        action,
        *,
        latent_frame_num=None,
        video_num_frames=None,
        state=None,
    ):
        if self.nmx_action_contract:
            if state is None:
                raise ValueError("NMX action contract requires an absolute state trajectory")
            raw_action, step_mask = prepare_raw_action_tensor(
                local_start_frame,
                latent_frame_ids,
                latent_frame_num,
                video_num_frames,
                action,
                self.config,
            )
            raw_state, _ = prepare_raw_action_tensor(
                local_start_frame,
                latent_frame_ids,
                latent_frame_num,
                video_num_frames,
                state,
                self.config,
            )
            actions, actions_mask = build_model_actions_from_raw(
                raw_action,
                raw_state,
                step_mask,
                self.config,
                q01=self.q01.squeeze(0),
                q99=self.q99.squeeze(0),
                chunk_size_frames=int(self.config.action_chunk_size_max),
            )
            return actions, actions_mask, raw_action, step_mask, raw_state

        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = np.clip(action_aligned, -1.5, 1.5)
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)
        ori_data_dict = self._clip_temporal_latents(ori_data_dict)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)

        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        out_dict = self._cat_video_latents(ori_data_dict)

        if self.nmx_action_contract:
            reference_key = self.used_video_keys[0]
            (
                out_dict['actions'],
                out_dict['actions_mask'],
                out_dict['raw_actions'],
                out_dict['raw_actions_step_mask'],
                out_dict['raw_states'],
            ) = self._action_post_process(
                local_start_frame,
                local_end_frame,
                latent_frame_ids,
                ori_data_dict['action'],
                latent_frame_num=int(
                    ori_data_dict[f"{reference_key}.latent_num_frames"]
                ),
                video_num_frames=int(
                    ori_data_dict[f"{reference_key}.video_num_frames"]
                ),
                state=ori_data_dict[self.state_column],
            )
            out_dict['action_q01'] = torch.from_numpy(self.q01.squeeze(0).copy())
            out_dict['action_q99'] = torch.from_numpy(self.q99.squeeze(0).copy())
            apply_episode_action_validity(
                out_dict,
                self.config,
                action_valid=bool(cur_meta.get('slam_validity_all', True)),
                width_valid=bool(cur_meta.get('width_validity', True)),
            )
        else:
            out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(
                local_start_frame,
                local_end_frame,
                latent_frame_ids,
                ori_data_dict['action'],
            )

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['demo_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
