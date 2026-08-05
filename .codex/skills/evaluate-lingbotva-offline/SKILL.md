---
name: evaluate-lingbotva-offline
description: Evaluate LingBot-VA IDM or FastWAM checkpoints over complete LeRobot episodes with cached training latents, teacher-forced history, matched action-chunk reconstruction, relative and absolute trajectory plots, golden NPZ output, and per-chunk inference timing. Use for checkpoint quality checks, IDM/FastWAM comparisons, K=1..4 evaluation, or generating a golden result for SuperInference process replay.
---

# Evaluate LingBot-VA Offline

Use `script/evaluate_nmx_offline.py`. Read
[references/offline-evaluation-contract.md](references/offline-evaluation-contract.md)
before choosing a checkpoint, config, dataset sample, or comparison metric.

## Workflow

1. Verify the checkout commit, checkpoint directory, base model, dataset, latent
   cache, action normalization stats, and CUDA environment.
2. Match the checkpoint to its training config. Use `nmx_chip_train` for baseline
   IDM and `nmx_chip_train_fastwam` for baseline FastWAM. Do not relabel an IDM
   checkpoint as FastWAM.
3. Recheck that the selected GPU is idle. Run both sides of a comparison on the
   same GPU without concurrent workloads.
4. Run the complete episode in `teacher_forced_video_full_episode` mode with the
   same dataset root, sample index, K, seed, and action-history mode.
5. Inspect `summary.json`, the relative plot, the absolute plot, and the NPZ.
6. Treat each model's NPZ as its own golden input for the corresponding
   SuperInference process replay.
7. Compare trajectory metrics and timing separately. Exclude the first chunk when
   reporting steady-state latency, and report whether video generation ran.

## Invocation

```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<PHYSICAL_GPU> \
  <PYTHON> script/evaluate_nmx_offline.py \
  --checkpoint <CHECKPOINT_STEP_DIR> \
  --base-model <LINGBOT_VA_BASE> \
  --dataset-root <LEROBOT_DATASET> \
  --config-name <nmx_chip_train|nmx_chip_train_fastwam> \
  --sample-index <GLOBAL_SAMPLE_INDEX> \
  --frame-chunk-size 4 \
  --evaluation-mode teacher_forced_video_full_episode \
  --action-history-mode zero \
  --seed 42 \
  --gpu 0 \
  --output-dir <NEW_OUTPUT_DIR>
```

When `CUDA_VISIBLE_DEVICES` contains one GPU, keep `--gpu 0`: it is the logical
device index inside that process. Always use a new output directory.

## Required Output

Require all of the following:

- `summary.json` reports the expected `action_condition_mode` and
  `video_run_mode`;
- the complete episode has the expected action count and chunk count;
- predictions and targets are finite;
- the NPZ contains relative predictions, targets, masks, chunk boundaries, and
  reconstructed absolute trajectories;
- both relative and absolute PNGs exist;
- per-chunk timing, total timing, first-chunk timing, and steady-state mean are
  present;
- IDM uses video generation; FastWAM action-only reports
  `video_run_mode=false`.

Do not call a checkpoint better solely because it is faster. Report action-quality
metrics, trajectory plots, and latency together.
