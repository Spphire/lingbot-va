# Offline Evaluation Contract

## Canonical Matched Comparison

Use the same values for both checkpoints:

- dataset: `/mnt/workspace/shenyibo/datasets/chip_0711_199episodes`
- global sample index: `109`
- evaluation mode: `teacher_forced_video_full_episode`
- frame chunk size: `4`
- action history: `zero`
- seed: `42`
- base model: `/mnt/workspace/shenyibo/lingbot-va-assets/lingbot-va-base`
- Python: `/mnt/workspace/shenyibo/lingbot-va-assets/venv-py312-torch271/bin/python`

The evaluator rebuilds relative-action targets for the requested K. It isolates
latent frame 0 as the causal condition, predicts later chunks, and then commits the
ground-truth video latent plus either zero or ground-truth action history into the KV
cache. For the production training runs here, dropout probability 1 means zero action
history is the matching contract.

## Checkpoint Pair

IDM baseline:

```text
/mnt/workspace/shenyibo/lingbot-va-fulltrains/
  chip_all_7858_visual_ab_dropah1_20260725_230557/baseline/
  checkpoints/checkpoint_step_20000
```

Use config `nmx_chip_train`. It must resolve to:

```text
action_condition_mode=inverse_dynamics
video_run_mode=true
```

FastWAM baseline:

```text
/mnt/workspace/shenyibo/lingbot-va-fulltrains/
  chip_all_7858_fastwam_visual_ab_dropah1_20260804_005600/baseline/
  checkpoints/checkpoint_step_20000
```

Use config `nmx_chip_train_fastwam`. It must resolve to:

```text
action_condition_mode=fastwam
video_run_mode=false
```

Do not edit a checkpoint's transformer `config.json` to switch modes. The training
config and checkpoint must remain paired.

## Artifacts And Interpretation

`sample_*.npz` is the numerical golden for deployment replay. `sample_*.png` shows
chunk-relative actions, while `sample_*_absolute.png` reconstructs world trajectories
from each chunk anchor. `summary.json` contains action quality and synchronized CUDA
timings.

Compare these quality metrics:

- left/right position RMSE in metres;
- left/right quaternion geodesic MAE in radians;
- overall masked MAE and RMSE;
- endpoint error where available.

For speed, report total `_infer` time, first-chunk time, and mean of chunks 2..N.
The first chunk is not a clean steady-state measurement. Run both models on the same
idle GPU, in the same environment, and record GPU identity and competing processes.

## Existing IDM Reference

The earlier step-20000 K=4 episode-109 run produced 756 actions in 16 chunks and
approximately:

```text
left position RMSE   0.008128 m
right position RMSE  0.007782 m
left rotation MAE    0.047748 rad
right rotation MAE   0.041817 rad
overall MAE          0.006304
```

Its output is under:

```text
/mnt/workspace/shenyibo/lingbot-va-fulltrains/
  chip_all_7858_visual_ab_dropah1_20260725_230557/baseline/
  offline_cache_fixed_step20000/chip_0711_episode109/k4
```

Regenerate both models with the current evaluator before making a timing comparison.

