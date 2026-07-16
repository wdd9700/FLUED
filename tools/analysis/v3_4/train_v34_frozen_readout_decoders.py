"""Fit two decoder forms against identical readouts from a frozen v3.4 encoder."""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flued.data import BYTE_VOCAB_SIZE, PAD_ID  # noqa: E402
from tools.train.v3_3.train_v33 import make_dataloaders, make_targets  # noqa: E402
from tools.train.v3_4.train_v34_pos_ar_probe import build_model  # noqa: E402


def _load_common(model, source: dict[str, torch.Tensor]) -> dict[str, list[str]]:
    target = model.state_dict()
    copied = {
        name: value
        for name, value in source.items()
        if name in target and target[name].shape == value.shape
    }
    target.update(copied)
    model.load_state_dict(target, strict=True)
    return {
        "copied": sorted(copied),
        "fresh": sorted(set(target) - set(copied)),
    }


def _decoder_params(model, mode: str) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if mode == "shared_inverse":
        modules = [model.interpreter_blocks, model.readout_pool, model.plain_byte_lookup]
    else:
        modules = [model.decoder]
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
            if id(parameter) not in seen:
                params.append(parameter)
                seen.add(id(parameter))
    return params


def _targets(token_ids, chunks, max_chunks: int, max_span: int):
    zero = torch.zeros_like(token_ids, dtype=torch.bool)
    targets, slot_mask, _ = make_targets(
        token_ids,
        zero,
        chunks.chunk_ids,
        chunks.offsets,
        max_chunks,
        max_span,
    )
    return targets, slot_mask


def _loss_and_acc(logits, targets, slot_mask):
    loss = F.cross_entropy(
        logits.float().reshape(-1, BYTE_VOCAB_SIZE),
        targets.reshape(-1),
        ignore_index=PAD_ID,
        reduction="none",
    ).view_as(targets)
    weight = slot_mask.to(loss.dtype)
    mean = (loss * weight).sum() / weight.sum().clamp(min=1.0)
    pred = logits.argmax(dim=-1)
    acc = ((pred == targets) & slot_mask).float().sum() / slot_mask.float().sum().clamp(min=1.0)
    return mean, acc


@torch.no_grad()
def _evaluate(encoder, decoders, loader, args, device, batches: int):
    encoder.eval()
    for decoder in decoders.values():
        decoder.eval()
    sums = {name: {"loss": 0.0, "acc": 0.0} for name in decoders}
    count = 0
    for index, batch in enumerate(loader):
        if index >= batches:
            break
        token_ids = batch[0].to(device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            encoded = encoder(token_ids)
            targets, slot_mask = _targets(token_ids, encoded.chunks, args.max_chunks, args.max_span)
            for name, decoder in decoders.items():
                logits = decoder.decode(
                    encoded.readout_z,
                    encoded.chunks.chunk_mask,
                    encoded.emit_hard,
                )
                loss, acc = _loss_and_acc(logits, targets, slot_mask)
                sums[name]["loss"] += float(loss.item())
                sums[name]["acc"] += float(acc.item())
        count += 1
    for values in sums.values():
        values["loss"] /= max(count, 1)
        values["acc"] /= max(count, 1)
    for decoder in decoders.values():
        decoder.train()
    return sums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--device", default="cuda")
    cli = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    checkpoint = Path(cli.checkpoint)
    config_path = Path(cli.config) if cli.config else checkpoint.with_name("resolved_config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({"device": cli.device, "num_workers": 0, "resume": False})
    args = Namespace(**config)
    device = torch.device(cli.device if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    source_state = payload["model"]

    encoder = build_model(args).to(device)
    encoder.load_state_dict(source_state, strict=True)
    encoder.requires_grad_(False).eval()

    shared_args = Namespace(**{**config, "decoder_mode": "shared_inverse"})
    independent_args = Namespace(**{**config, "decoder_mode": "legacy_independent"})
    shared = build_model(shared_args).to(device)
    shared.load_state_dict(source_state, strict=True)
    independent = build_model(independent_args).to(device)
    load_info = _load_common(independent, source_state)

    shared_params = _decoder_params(shared, "shared_inverse")
    independent_params = _decoder_params(independent, "legacy_independent")
    decoders = {"shared_inverse": shared, "independent": independent}
    trainable_params = {
        "shared_inverse": shared_params,
        "independent": independent_params,
    }
    optimizers = {
        "shared_inverse": torch.optim.AdamW(shared_params, lr=cli.lr, weight_decay=0.01, fused=device.type == "cuda"),
        "independent": torch.optim.AdamW(independent_params, lr=cli.lr, weight_decay=0.01, fused=device.type == "cuda"),
    }

    train_loader, eval_loader = make_dataloaders(args)
    train_iter = iter(train_loader)
    rows = []
    start = time.perf_counter()

    def record(step: int):
        metrics = _evaluate(encoder, decoders, eval_loader, args, device, cli.eval_batches)
        row = {"step": step, "elapsed_sec": time.perf_counter() - start, **metrics}
        rows.append(row)
        print(
            f"[eval {step}] shared={metrics['shared_inverse']['loss']:.4f}/"
            f"{metrics['shared_inverse']['acc']:.4f} independent="
            f"{metrics['independent']['loss']:.4f}/{metrics['independent']['acc']:.4f}",
            flush=True,
        )

    record(0)
    for step in range(1, cli.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        token_ids = batch[0].to(device, non_blocking=device.type == "cuda")
        with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            encoded = encoder(token_ids)
            readout = encoded.readout_z.detach()
            targets, slot_mask = _targets(token_ids, encoded.chunks, args.max_chunks, args.max_span)

        for name, decoder in decoders.items():
            optimizers[name].zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
            ):
                logits = decoder.decode(
                    readout,
                    encoded.chunks.chunk_mask,
                    encoded.emit_hard,
                )
                loss, _ = _loss_and_acc(logits, targets, slot_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params[name], 1.0)
            optimizers[name].step()

        if step % cli.eval_every == 0 or step == cli.steps:
            record(step)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "steps": cli.steps,
        "lr": cli.lr,
        "shared_trainable_params": sum(p.numel() for p in shared_params),
        "independent_trainable_params": sum(p.numel() for p in independent_params),
        "independent_load_info": load_info,
        "curve": rows,
    }
    (out_dir / "frozen_readout_decoder_probe.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    last = rows[-1]
    markdown = [
        "# 冻结 readout 的 decoder 归因实验",
        "",
        f"来源检查点：`{checkpoint}`",
        "",
        f"共享逆可训练参数：{result['shared_trainable_params']:,}",
        f"独立 decoder 可训练参数：{result['independent_trainable_params']:,}",
        "",
        "| 步数 | 共享逆损失 | 共享逆准确率 | 独立损失 | 独立准确率 |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        markdown.append(
            f"| {row['step']} | {row['shared_inverse']['loss']:.4f} | "
            f"{row['shared_inverse']['acc']:.4f} | {row['independent']['loss']:.4f} | "
            f"{row['independent']['acc']:.4f} |"
        )
    markdown.extend(
        [
            "",
            "最终损失差："
            f"{last['shared_inverse']['loss'] - last['independent']['loss']:+.4f}（共享逆减独立）。",
            "",
        ]
    )
    (out_dir / "frozen_readout_decoder_probe.md").write_text("\n".join(markdown), encoding="utf-8")


if __name__ == "__main__":
    main()
