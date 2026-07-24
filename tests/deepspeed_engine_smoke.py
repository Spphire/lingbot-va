import argparse
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist

from wan_va.distributed.deepspeed import materialize_deepspeed_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument(
        "--config", default="config/deepspeed/zero2.json"
    )
    parser.add_argument(
        "--output", default="/tmp/lingbot_va_deepspeed_engine_smoke"
    )
    args = parser.parse_args()

    import deepspeed

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    torch.manual_seed(1234)

    model = torch.nn.Linear(8, 4).to(
        device=f"cuda:{local_rank}", dtype=torch.bfloat16
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    ds_config = materialize_deepspeed_config(
        args.config,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        world_size=world_size,
        param_dtype=torch.bfloat16,
        gradient_clipping=1.0,
    )
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )

    for micro_step in range(4):
        generator = torch.Generator(device=engine.device)
        generator.manual_seed(1000 + dist.get_rank() * 10 + micro_step)
        inputs = torch.randn(
            2,
            8,
            generator=generator,
            device=engine.device,
            dtype=torch.bfloat16,
        )
        loss = engine(inputs).float().square().mean()
        engine.backward(loss)
        engine.step()

    if int(engine.global_steps) != 2:
        raise AssertionError(f"expected 2 optimizer steps, got {engine.global_steps}")

    checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in engine.module.parameters()]
    ).sum()
    minimum = checksum.clone()
    maximum = checksum.clone()
    dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    if not torch.equal(minimum, maximum):
        raise AssertionError(
            f"parameters diverged across ranks: min={minimum.item()} max={maximum.item()}"
        )

    output = Path(args.output)
    if dist.get_rank() == 0:
        shutil.rmtree(output, ignore_errors=True)
    dist.barrier(device_ids=[local_rank])
    engine.save_checkpoint(
        str(output), tag="smoke", client_state={"step": int(engine.global_steps)}
    )
    load_path, client_state = engine.load_checkpoint(str(output), tag="smoke")
    if load_path is None or int((client_state or {}).get("step", -1)) != 2:
        raise AssertionError("DeepSpeed checkpoint round-trip failed")
    dist.barrier(device_ids=[local_rank])
    if dist.get_rank() == 0:
        print("DeepSpeed ZeRO-2 smoke passed on 2 GPUs")
        shutil.rmtree(output, ignore_errors=True)
    dist.barrier(device_ids=[local_rank])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
