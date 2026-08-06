# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Integrated into nmx_vla by Wendi Chen.
"""Shared missing_action_filling channel-wise action embedder helpers.

This module holds the missing-action-filling unit extracted verbatim from
``model.py`` so the core model file (``model.py``) can stay focused on the
transformer class itself. Every helper here is self-contained: each operates on
a ``module`` (or on raw tensors/weights) passed in as an argument, so NONE of
them imports the model classes — which is what lets BOTH backbones use the same
implementation without a circular import:

* ``WanTransformer3DModel`` (shared backbone, ``model.py``)
* ``WanTransformer3DMoTModel`` (MoT backbone, ``model_mot.py``)

The missing_action_filling ("missing action filling") feature gives missing / invalid action
channels their own learned embedding column instead of letting them silently
train as zero padding. The unit covers: per-frame validity alignment, YAML-arg
coercion, the dedicated checkpoint-mismatch exception, the checkpoint config
verification, the per-frame fail-loud guard, the core v/(1-v) embedding math, and
the __init__-time registration of the missing_action_filling parameter + config buffers.

IMPORTANT: this module must NEVER import ``.model`` or ``.model_mot`` — doing so
would create a circular import. The helpers are deliberately model-class-free.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)


def _align_per_frame_validity_to_length(
    per_frame_validity, batch_size, sequence_length, *, stream_name
):
    """Reshape a per-frame validity stream to [B, L]; RAISE on any length mismatch.

    The dataloader emits per-frame validity already aligned to the model's flattened
    (frame x sub-frame) sequence, so its length MUST equal `sequence_length`. Any mismatch
    (longer OR shorter) is an upstream alignment/export bug, not a recoverable runtime
    state, so we raise immediately. Crashing on the first step is far better than silently
    training on misaligned supervision for thousands of steps — a warning buried in a
    multi-hour training log is effectively invisible, and the previous code even truncated
    the longer-than-L case with no signal at all.

    Args:
        per_frame_validity: bool/float tensor broadcastable to [B, F, N] or [B, L],
            already moved onto the compute device by the caller.
        batch_size: B, the batch dimension of the action stream.
        sequence_length: L, the flattened (frame x sub-frame) target sequence length.
        stream_name: "action" or "width" — used in the error message.

    Returns:
        float tensor [B, L] of 0./1. validity with length exactly `sequence_length`.

    Raises:
        ValueError: if the flattened validity length != `sequence_length` — an upstream
            dataloader alignment/export bug that must be fixed, not masked.
    """
    validity = per_frame_validity.bool().float().reshape(batch_size, -1)
    current_length = validity.shape[1]
    if current_length != sequence_length:
        raise ValueError(
            f"per-frame {stream_name} validity length {current_length} != expected "
            f"sequence length {sequence_length}. This is an upstream dataloader "
            f"alignment/export bug — fix the per-frame validity alignment rather than "
            f"masking it, so misaligned supervision can never silently reach the loss."
        )
    return validity


def _resolve_int_list_arg(arg_value):
    """Coerce a yaml-plumbed missing_action_filling int-list knob to list[int].

    The knob flows `datasets.vla_data.learnable_action_embedding.*` yaml -> runtime_config ->
    `from_pretrained` kwarg.
    YAML is the ONLY source — no env-var fallback: missing_action_filling is a new feature, so there is
    no legacy launcher exporting old env vars to stay compatible with. None (knob omitted
    from yaml) -> [] (that knob is off).

    Args:
        arg_value: iterable of channel indices, or None.

    Returns:
        list[int], possibly empty.
    """
    if arg_value is None:
        return []
    return [int(x) for x in arg_value]


def _resolve_bool_arg(arg_value):
    """Coerce a yaml-plumbed missing_action_filling bool knob to bool. None -> False. YAML-only (no env).

    Args:
        arg_value: bool, or None (knob omitted from yaml).

    Returns:
        bool.
    """
    return bool(arg_value) if arg_value is not None else False


class MissingActionFillingConfigMismatchError(Exception):
    """Raised when a checkpoint's missing_action_filling config does not match this run's.

    WHY a DEDICATED exception class (not RuntimeError): the MoT load path in
    ``loaders.py:load_transformer`` wraps ``WanTransformer3DMoTModel.from_pretrained`` in a wide
    ``except (OSError, ValueError, RuntimeError)`` that falls back to loading from a shared model.
    A missing_action_filling mismatch raised as RuntimeError gets SWALLOWED by that except -> the run silently
    falls back to shared, hiding a real column-mapping error. This class inherits directly from
    ``Exception`` and is intentionally NOT in that except tuple, so a genuine config mismatch
    propagates out loudly instead of degrading into a silent shared-model fallback.
    """


def _check_missing_action_filling_config_buf(module, state_dict, prefix):
    """Verify a checkpoint's saved missing_action_filling config matches this run's, mutating state_dict.

    Shared by BOTH the shared-backbone (WanTransformer3DModel) and MoT
    (WanTransformer3DMoTModel) ``_load_from_state_dict`` overrides so the two can't drift apart.
    The dataset-level mask config is saved in checkpoint buffers; we compare it
    against the module's live config and raise
    ``MissingActionFillingConfigMismatchError`` on mismatch. Old checkpoints
    without a given buffer get a warning + a synthesized buffer so they still load.

    WHY extracted to a module-level helper: MoT previously had NO such check,
    while the shared backbone did. One helper, two callers, no second copy to drift.

    Args:
        module: the nn.Module being loaded (reads missing_action_filling config buffers).
        state_dict: the incoming checkpoint dict (mutated in place for old-ckpt back-fill).
        prefix: the module's key prefix within ``state_dict``.
    """
    import warnings

    # Each checked config entry: (buffer attr name, list-valued live config) — the buffer holds
    # the value the checkpoint was trained with; the live value is what THIS run configured.
    # Mismatched used/gripper/pad_unused changes the validity mask semantics, so
    # it should not load silently.
    _checked_entries = [
        (
            "_missing_action_filling_used_buf",
            [
                int(x)
                for x in getattr(module, "_missing_action_filling_used_channels", [])
            ],
        ),
        (
            "_missing_action_filling_gripper_buf",
            [
                int(x)
                for x in getattr(module, "_missing_action_filling_gripper_channels", [])
            ],
        ),
        # pad_unused is a bool; store it as a single-element 0/1 list so the compare is uniform.
        (
            "_missing_action_filling_pad_unused_buf",
            [int(bool(getattr(module, "_missing_action_filling_pad_unused", False)))],
        ),
    ]
    for buf_attr, cur_value in _checked_entries:
        # The buffer key as it appears in the checkpoint dict for THIS module.
        key = prefix + buf_attr
        if key in state_dict:
            # Checkpoint carries the value it was trained with -> compare exactly.
            ckpt_value = [int(x) for x in state_dict[key].tolist()]
            if ckpt_value != cur_value:
                # Mismatch means the column->channel mapping would differ -> fail loudly with the
                # DEDICATED exception so the MoT loader's broad except can't swallow it.
                raise MissingActionFillingConfigMismatchError(
                    f"missing_action_filling config mismatch on {buf_attr}: checkpoint trained with "
                    f"{ckpt_value} but this run configured {cur_value}. Match "
                    f"learnable_action_embedding dataset_used_canonical_dims / "
                    f"gripper_canonical_dims / enable_unused_canonical_dims to the "
                    f"training run."
                )
        elif hasattr(module, buf_attr):
            # Old checkpoint predates this buffer: can't verify, so warn and back-fill the current
            # buffer into state_dict so the load still succeeds.
            warnings.warn(
                f"checkpoint predates {buf_attr}; cannot verify missing_action_filling config "
                f"consistency — ensure missing_action_filling settings match the training run.",
                stacklevel=2,
            )
            state_dict[key] = getattr(module, buf_attr).clone()


def _assert_missing_action_filling_configured_for_per_frame(
    replace_param, *per_frame_validities
):
    """Fail loudly when per-frame validity carries real invalidity but missing_action_filling is unconfigured.

    Per-frame / per-channel validity can only be honored by the missing_action_filling path.
    This guard remains for older/non-resident model variants where the parameter may
    be absent.

    WHY the check is "contains any invalid" rather than "validity key is present": the collate path makes the
    dataset emit these per-frame keys UNCONDITIONALLY (clean samples get an all-valid placeholder
    grid) so ``default_collate`` never sees a heterogeneous batch dict. An all-valid placeholder
    carries NO missing data, so a clean run (no missing_action_filling) must NOT be rejected by it. We therefore
    only raise when a validity tensor actually contains a value marking something invalid (< 0.5).

    Args:
        replace_param: the model's missing_action_filling nn.Parameter.
        *per_frame_validities: any number of per-frame validity tensors (or None) to inspect.
    """
    # missing_action_filling configured -> the embed path handles per-frame validity, nothing to guard.
    if replace_param is not None:
        return
    for validity in per_frame_validities:
        if validity is None:
            continue
        # Any entry < 0.5 means at least one (frame[, channel]) is marked invalid -> the missing_action_filling
        # path is required to honor it, but it isn't configured -> fail loudly.
        if bool((validity.float() < 0.5).any()):
            shapes = [
                tuple(v.shape) if v is not None else None for v in per_frame_validities
            ]
            message = (
                "received per-frame validity marking data invalid, but missing_action_filling is not "
                "configured (the legacy null_action_tokens path is episode-level only). Set "
                "datasets.vla_data.learnable_action_embedding.enabled=true with "
                "dataset_used_canonical_dims / gripper_canonical_dims / enable_unused_canonical_dims "
                f"as needed, or use an episode-level dataset. validity_shapes={shapes}"
            )
            logger.error("%s", message)
            raise RuntimeError(message)


def _embed_action_with_missing_filling(
    action_input,
    action_validity,
    width_validity,
    action_validity_per_frame,
    width_validity_per_frame,
    channel_validity_per_frame,
    missing_action_filling_enabled,
    *,
    w_normal_weight,
    w_normal_bias,
    w_missing_param,
    used_channels,
    gripper_channels,
    pad_unused,
    used_channel_mask=None,
    gripper_channel_mask=None,
    pad_unused_mask=None,
):
    """Shared core of missing_action_filling action embedding for BOTH backbones (shared + MoT).

    WHY: model.py (WanTransformer3DModel) and model_mot.py (WanTransformer3DMoTModel)
    had two byte-for-byte-equivalent copies of the validity-grid build + v/(1-v) split +
    ``(raw*v)@W_normal + (1-v)@missing_action_filling + bias`` matmul. We collapse that
    redundancy into ONE implementation so the math can't drift between the two classes.
    The ONLY real differences between the two were (a) which linear's weight/bias is used
    (shared: ``action_embedder``; MoT: ``action_encoder[0]``) and (b) MoT runs the result
    through ``action_encoder[1:]`` (GELU + 2nd linear) afterwards — so the weights/bias/
    replace-param are passed in as args, and the post-linear stack stays in the MoT wrapper.

    Computes: ``action_h = (raw * v) @ W_normal.T + (1 - v) @ missing_action_filling.T + bias``,
    where v is the per-sample/per-frame/per-channel validity mask (1 = present,
    0 = missing). The mask is built from action+width validity, optional
    full channel-validity grids, and per-sample dataset layout masks.

    Args:
        action_input: ``[B, action_dim, F, H, W]`` raw action.
        action_validity / width_validity: per-sample bool ``[B]`` for used / gripper channels
            (None means all valid for that stream).
        action_validity_per_frame / width_validity_per_frame: optional ``[B, F, N]`` per-frame
            validity; used instead of the per-sample flag when present.
        channel_validity_per_frame: optional ``[B, F, N, action_dim]`` grid; AND-ed into the mask.
        w_normal_weight / w_normal_bias: the normal linear's weight ``[out, action_dim]`` / bias.
        w_missing_param: the learned missing_action_filling parameter ``[out, action_dim]``.
        used_channels / gripper_channels: fallback canonical channel index lists masked by action / width.
        pad_unused: when True, every channel not in used_channels is forced missing (v=0).
        used_channel_mask / gripper_channel_mask / pad_unused_mask: optional
            per-sample overrides from the dataloader. These are required for
            heterogeneous dataset mixtures where different datasets occupy
            different slices of the same canonical action space.

    Returns:
        ``[B, L, out]`` the normal-plus-missing linear output (pre any further encoder layers).
    """
    # Cast raw to the normal weight's dtype so the rearrange/matmul dtype is consistent
    # with the embedder weight (mirrors both original bodies' .to(weight.dtype)).
    raw = rearrange(action_input, "b c f h w -> b (f h w) c").to(w_normal_weight.dtype)
    B, L, D = raw.shape

    def _resolve_channel_mask(mask_arg, channel_list, *, mask_name):
        if mask_arg is None:
            mask = torch.zeros(B, D, device=raw.device, dtype=torch.bool)
            for channel in channel_list or []:
                mask[:, int(channel)] = True
            return mask.float()

        mask = torch.as_tensor(mask_arg, device=raw.device).bool()
        if mask.dim() == 1:
            if mask.shape[0] != D:
                raise ValueError(
                    f"{mask_name} must have width action_dim={D}, got {tuple(mask.shape)}"
                )
            mask = mask.unsqueeze(0).expand(B, -1)
        else:
            mask = mask.reshape(B, -1)
            if mask.shape != (B, D):
                raise ValueError(
                    f"{mask_name} must have shape [B, action_dim]=[{B}, {D}], "
                    f"got {tuple(mask.shape)}"
                )
        return mask.float()

    used_mask = _resolve_channel_mask(
        used_channel_mask,
        used_channels,
        mask_name="learnable_action_embedding_used_mask",
    )
    gripper_mask = _resolve_channel_mask(
        gripper_channel_mask,
        gripper_channels,
        mask_name="learnable_action_embedding_gripper_mask",
    )

    if pad_unused_mask is None:
        pad_unused_b = torch.full(
            (B,),
            bool(pad_unused),
            device=raw.device,
            dtype=torch.bool,
        )
    else:
        pad_unused_b = torch.as_tensor(pad_unused_mask, device=raw.device).bool()
        if pad_unused_b.dim() == 0:
            pad_unused_b = pad_unused_b.expand(B)
        else:
            pad_unused_b = pad_unused_b.reshape(B)

    # Build the per-channel validity mask v [B, L, action_dim]. Start all-valid, then
    # multiply in each "missing" rule. Per-frame validity when the dataset provides it,
    # else the per-sample bool broadcast over all frames. Parameter registration is
    # unconditional, so per-sample enabled decides whether validity streams are
    # allowed to affect a sample at all.
    v = torch.ones(B, L, D, device=raw.device, dtype=torch.float32)
    if missing_action_filling_enabled is None:
        enabled_b = torch.ones(B, device=raw.device, dtype=torch.float32)
    else:
        enabled_b = torch.as_tensor(
            missing_action_filling_enabled,
            device=raw.device,
        ).bool()
        enabled_b = enabled_b.float().reshape(B)
    enabled_bl = enabled_b[:, None]
    enabled_bld = enabled_b[:, None, None]
    used_bld = used_mask[:, None, :]
    gripper_bld = gripper_mask[:, None, :]

    # Action validity: used_channels are masked where the action stream is invalid.
    if action_validity_per_frame is not None:
        action_validity_flat = _align_per_frame_validity_to_length(
            action_validity_per_frame.to(raw.device), B, L, stream_name="action"
        )
        action_validity_flat = 1.0 - enabled_bl + enabled_bl * action_validity_flat
        v = v * (1.0 - used_bld + used_bld * action_validity_flat[:, :, None])
    elif action_validity is not None:
        action_v = action_validity.to(raw.device).bool().float()
        action_v = 1.0 - enabled_b + enabled_b * action_v
        v = v * (1.0 - used_bld + used_bld * action_v[:, None, None])

    # Width validity: gripper_channels are masked where the gripper is invalid.
    if width_validity_per_frame is not None:
        width_validity_flat = _align_per_frame_validity_to_length(
            width_validity_per_frame.to(raw.device), B, L, stream_name="width"
        )
        width_validity_flat = 1.0 - enabled_bl + enabled_bl * width_validity_flat
        v = v * (1.0 - gripper_bld + gripper_bld * width_validity_flat[:, :, None])
    elif width_validity is not None:
        width_v = width_validity.to(raw.device).bool().float()
        width_v = 1.0 - enabled_b + enabled_b * width_v
        v = v * (1.0 - gripper_bld + gripper_bld * width_v[:, None, None])

    # Pad-unused: mark every channel we don't use as missing (v=0) so it goes through
    # missing_action_filling instead of staying as zero padding.
    pad_unused_active = (pad_unused_b.float() * enabled_b)[:, None, None]
    v = v * (1.0 - pad_unused_active * (1.0 - used_bld))

    # If the dataset gives a full per-channel-per-frame validity grid, AND it into v.
    # validate rank + last-dim BEFORE reshape. A grid of the wrong dim order or width
    # (e.g. [B, F, N, C_used] with C_used != action_dim) can still flatten to a length that
    # happens to equal L, silently mis-mapping channels. So we require the last dim to be the
    # action_dim D and the tensor to have at least 2 dims before trusting the reshape.
    if channel_validity_per_frame is not None:
        grid_raw = channel_validity_per_frame.to(raw.device).float()
        if grid_raw.dim() < 2 or grid_raw.shape[-1] != D:
            raise ValueError(
                f"channel_validity_per_frame has shape {tuple(grid_raw.shape)}; expected its "
                f"last dim to be action_dim={D} and rank>=2. A wrong dim order/width can "
                f"still flatten to the right length and silently mis-map channels — fix the "
                f"upstream grid shape."
            )
        grid = grid_raw.reshape(B, -1, D)
        if grid.shape[1] != L:
            raise ValueError(
                f"channel_validity_per_frame length {grid.shape[1]} != expected "
                f"sequence length {L}. This is an upstream dataloader alignment/export "
                f"bug — fix the alignment rather than masking it, so misaligned "
                f"supervision can never silently reach the loss."
            )
        # AND the grid into v (multiplicative) instead of OVERWRITING v. Overwriting
        # (v = grid) discarded the per-sample action/width validity computed above whenever a
        # channel grid was present, so a sample whose whole action stream was invalid could be
        # un-masked by an all-valid channel grid. AND keeps both signals: a cell is valid only
        # if BOTH the action/width masks and the channel grid say so.
        grid = 1.0 - enabled_bld + enabled_bld * grid
        v = v * grid

    # Compute in fp32 for stability.
    raw_f = raw.float()
    # Present channels (raw * v) go through the normal weight.
    valid_contrib = F.linear(raw_f * v, w_normal_weight.float(), bias=None)
    # Missing channels (1 - v) go through the full-width learned missing-action
    # parameter. Per-sample/per-channel v, not a static config list, decides
    # which channels contribute on each sample.
    if w_missing_param is not None:
        replace_contrib = F.linear(1.0 - v, w_missing_param.float(), bias=None)
    else:
        # Older/non-resident model variant: nothing to add.
        replace_contrib = torch.zeros_like(valid_contrib)
    bias = w_normal_bias.float()
    # Output in raw dtype; caller (MoT) may run further encoder layers on top.
    return (valid_contrib + replace_contrib + bias).to(raw.dtype)


def _make_action_condition_all_missing_inputs(action_input):
    """Build the inference-time equivalent of train-time action-history dropout.

    action_history_condition_dropout_prob=1 trains the condition branch with a
    zero channel-validity grid and learnable_action_embedding_enabled=True. The
    raw action values are ignored by the missing-action embedder once validity is
    zero everywhere; this helper supplies the two tensors that enforce that path.
    """
    batch_size = int(action_input.shape[0])
    action_dim = int(action_input.shape[1])
    action_token_count = int(
        action_input.shape[-3] * action_input.shape[-2] * action_input.shape[-1]
    )
    channel_validity_all_missing = torch.zeros(
        batch_size,
        action_token_count,
        action_dim,
        device=action_input.device,
        dtype=torch.float32,
    )
    missing_action_filling_enabled = torch.ones(
        batch_size,
        device=action_input.device,
        dtype=torch.bool,
    )
    return channel_validity_all_missing, missing_action_filling_enabled


def _register_missing_action_filling(
    module,
    *,
    used_channels_arg,
    gripper_channels_arg,
    pad_unused_arg,
    action_dim,
    hidden_dim,
    param_name,
):
    """Resolve + register the missing_action_filling config on a module (shared OR MoT __init__).

    WHY: model.py and model_mot.py __init__ each held a byte-for-byte-equivalent block that
    resolved the YAML learnable_action_embedding.* args, bounds-checked them,
    registered the missing_action_filling nn.Parameter, and registered the config
    buffers. We collapse that redundancy. The ONLY differences were action_dim
    (shared: action_embedder.weight.shape[1]; MoT: action_encoder[0].weight.shape[1]),
    hidden_dim (shared: inner_dim; MoT: action_encoder[0].weight.shape[0]), and the param
    name (action_embedder_missing vs action_encoder_missing) — all passed in as args here.

    Sets on ``module``: ``_missing_action_filling_used_channels``,
    ``_missing_action_filling_gripper_channels``, ``_missing_action_filling_pad_unused``,
    the ``<param_name>`` parameter (an ``nn.Parameter`` ``[hidden_dim, action_dim]``), and
    the ``_missing_action_filling_*_buf`` long config buffers.

    Args:
        module: the nn.Module (the transformer) to register parameters/buffers on.
        used_channels_arg / gripper_channels_arg / pad_unused_arg: the raw
            YAML learnable_action_embedding.* args (resolved here via _resolve_*).
        action_dim: canonical channel count (= the normal linear's input width). Used for the
            the all-channels default and the bounds check.
        hidden_dim: the missing_action_filling parameter's output width (the normal linear's output width).
        param_name: attribute name to register the missing_action_filling parameter under.
    """
    # The parameter is always present. Dataset-level enabled flags decide whether
    # a sample produces a non-trivial missing mask; keeping the parameter resident
    # avoids config-order-dependent model shapes in heterogeneous mixtures.
    module._missing_action_filling_used_channels = _resolve_int_list_arg(
        used_channels_arg
    )
    module._missing_action_filling_gripper_channels = _resolve_int_list_arg(
        gripper_channels_arg
    )
    # fail fast on any out-of-range channel index so a typo'd yaml can't silently point a
    # missing_action_filling column / mask at a nonexistent channel (or wrap a negative index).
    for _name, _channels in (
        (
            "learnable_action_embedding.dataset_used_canonical_dims",
            module._missing_action_filling_used_channels,
        ),
        (
            "learnable_action_embedding.gripper_canonical_dims",
            module._missing_action_filling_gripper_channels,
        ),
    ):
        for _c in _channels:
            if not (0 <= _c < action_dim):
                raise ValueError(
                    f"{_name} channel index {_c} out of range [0, {action_dim}); "
                    f"action_dim is {action_dim}."
                )
    # Pad-unused: instead of leaving the canonical channels we don't use as plain zeros, give
    # each its own learned missing_action_filling column. Off by default, so existing configs are unchanged.
    module._missing_action_filling_pad_unused = _resolve_bool_arg(pad_unused_arg)
    module.register_parameter(
        param_name,
        nn.Parameter(torch.randn(hidden_dim, action_dim) * 0.02),
    )
    # Save the missing_action_filling config into the checkpoint. If a later
    # eval/resume uses a different dataset-level layout, the mask semantics would
    # differ; the _load_from_state_dict check turns this into a loud error.
    # used/gripper are long buffers; pad_unused is a
    # 0/1 long buffer so the same tensor-compare path handles it.
    module.register_buffer(
        "_missing_action_filling_used_buf",
        torch.tensor(
            list(module._missing_action_filling_used_channels), dtype=torch.long
        ),
        persistent=True,
    )
    module.register_buffer(
        "_missing_action_filling_gripper_buf",
        torch.tensor(
            list(module._missing_action_filling_gripper_channels), dtype=torch.long
        ),
        persistent=True,
    )
    module.register_buffer(
        "_missing_action_filling_pad_unused_buf",
        torch.tensor(
            [int(bool(module._missing_action_filling_pad_unused))], dtype=torch.long
        ),
        persistent=True,
    )


__all__ = [
    "MissingActionFillingConfigMismatchError",
    "_align_per_frame_validity_to_length",
    "_resolve_int_list_arg",
    "_resolve_bool_arg",
    "_check_missing_action_filling_config_buf",
    "_assert_missing_action_filling_configured_for_per_frame",
    "_embed_action_with_missing_filling",
    "_make_action_condition_all_missing_inputs",
    "_register_missing_action_filling",
]
