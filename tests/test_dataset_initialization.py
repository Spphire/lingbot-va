from types import SimpleNamespace

from wan_va.dataset import lerobot_latent_dataset as dataset_module


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
