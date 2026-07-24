from types import SimpleNamespace

from datasets.features import features as feature_module

from wan_va.dataset import lerobot_latent_dataset as dataset_module


def test_registers_list_schema_compatibility(monkeypatch):
    feature_types = {}
    monkeypatch.setattr(feature_module, "_FEATURE_TYPES", feature_types)

    dataset_module.ensure_hf_datasets_list_compat()

    assert feature_types["List"] is feature_module.Sequence


def test_single_dataset_is_constructed_without_worker_pool(monkeypatch):
    config = SimpleNamespace(dataset_path="/datasets")
    calls = []

    monkeypatch.setattr(
        dataset_module,
        "recursive_find_file",
        lambda *_args: ["/datasets/official/meta/info.json"],
    )
    monkeypatch.setattr(
        dataset_module,
        "construct_lerobot",
        lambda repo_id, config: calls.append((repo_id, config)) or repo_id,
    )

    class UnexpectedExecutor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("single-dataset setup must not create workers")

    monkeypatch.setattr(dataset_module, "ThreadPoolExecutor", UnexpectedExecutor)

    datasets = dataset_module.construct_lerobot_multi_processor(
        config,
        num_init_worker=8,
    )

    assert datasets == ["/datasets/official"]
    assert calls == [("/datasets/official", config)]
