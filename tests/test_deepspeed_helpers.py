import json
import tempfile
import unittest
from pathlib import Path

import torch

from wan_va.distributed.deepspeed import (
    materialize_deepspeed_config,
    prepare_accumulated_losses,
    validate_distributed_backend,
)


class DeepSpeedHelpersTest(unittest.TestCase):
    def test_materializes_auto_values(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "zero2.json"
            path.write_text(
                json.dumps(
                    {
                        "train_batch_size": "auto",
                        "train_micro_batch_size_per_gpu": "auto",
                        "gradient_accumulation_steps": "auto",
                        "gradient_clipping": "auto",
                        "bf16": {"enabled": "auto"},
                        "fp16": {"enabled": "auto"},
                    }
                ),
                encoding="utf-8",
            )
            config = materialize_deepspeed_config(
                path,
                micro_batch_size=2,
                gradient_accumulation_steps=4,
                world_size=16,
                param_dtype=torch.bfloat16,
                gradient_clipping=2.0,
            )

        self.assertEqual(config["train_batch_size"], 128)
        self.assertEqual(config["train_micro_batch_size_per_gpu"], 2)
        self.assertEqual(config["gradient_accumulation_steps"], 4)
        self.assertEqual(config["gradient_clipping"], 2.0)
        self.assertTrue(config["bf16"]["enabled"])
        self.assertFalse(config["fp16"]["enabled"])

    def test_deepspeed_owns_backward_accumulation_scaling(self):
        payload = prepare_accumulated_losses(
            torch.tensor(8.0),
            torch.tensor(4.0),
            gradient_accumulation_steps=4,
            uses_deepspeed=True,
        )
        self.assertEqual(payload["backward_loss"].item(), 12.0)
        self.assertEqual(payload["latent_loss_for_log"].item(), 2.0)
        self.assertEqual(payload["action_loss_for_log"].item(), 1.0)

    def test_fsdp_scales_backward_loss(self):
        payload = prepare_accumulated_losses(
            torch.tensor(8.0),
            torch.tensor(4.0),
            gradient_accumulation_steps=4,
            uses_deepspeed=False,
        )
        self.assertEqual(payload["backward_loss"].item(), 3.0)

    def test_backend_validation(self):
        self.assertEqual(validate_distributed_backend(" DeepSpeed "), "deepspeed")
        with self.assertRaises(ValueError):
            validate_distributed_backend("ddp")


if __name__ == "__main__":
    unittest.main()
