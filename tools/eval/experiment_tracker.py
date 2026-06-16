# experiment_tracker.py — Unified metadata recorder for all FLUED experiments
#
# Usage:
#   python experiment_tracker.py record --exp D1_BPB --model flued_v2 \
#       --train-corpus corpus_v3.txt --eval-corpus corpus_v3_test.txt \
#       --train-steps 50000 --batch-size 2 --seq-len 512 \
#       --params 328M --wall-time 13.5h --throughput 12345 \
#       --bpb 0.876 --notes "v2 seed=42, frozen encoder"
#
#   python experiment_tracker.py table  → prints markdown table of all experiments

import argparse, json, os, sys
from datetime import datetime
from pathlib import Path

TRACKER_PATH = "checkpoints/experiment_tracker.json"

SCHEMA = {
    "exp_id":       "",     # e.g. "D1_BPB_flued_v2"
    "phase":        "",     # A / D0 / D1 / AB / E3
    "model":        "",     # flued_v2 / blt / bpe_8k / bpe_16k / bpe_32k / byte
    "train_corpus": "",
    "eval_corpus":  "",
    "train_steps":  0,
    "effective_batch": 0,
    "seq_len":      0,
    "params":       "",     # e.g. "328M"
    "wall_time_h":  0.0,
    "throughput_bytes_s": 0,
    "final_bpb":    None,
    "final_acc":    None,
    "notes":        "",
    "date":         "",
}

def load():
    if os.path.exists(TRACKER_PATH):
        with open(TRACKER_PATH) as f:
            return json.load(f)
    return []

def save(records):
    with open(TRACKER_PATH, 'w') as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

def cmd_record(args):
    records = load()
    rec = dict(SCHEMA)
    rec["exp_id"] = args.exp
    rec["phase"] = args.phase
    rec["model"] = args.model
    rec["train_corpus"] = args.train_corpus
    rec["eval_corpus"] = args.eval_corpus
    rec["train_steps"] = args.train_steps
    rec["effective_batch"] = args.effective_batch
    rec["seq_len"] = args.seq_len
    rec["params"] = args.params
    rec["wall_time_h"] = args.wall_time
    rec["throughput_bytes_s"] = args.throughput
    rec["final_bpb"] = args.bpb
    rec["final_acc"] = args.acc
    rec["notes"] = args.notes
    rec["date"] = datetime.now().isoformat()
    records.append(rec)
    save(records)
    print(f"Recorded: {rec['exp_id']}")

def cmd_table(args):
    records = load()
    if not records:
        print("No experiments recorded.")
        return
    print(f"\n{'Phase':>5} | {'Model':>12} | {'Steps':>7} | {'Batch':>5} | {'Seq':>4} | {'Params':>7} | {'Wall(h)':>7} | {'Bytes/s':>7} | {'BPB':>6} | {'Acc':>6}")
    print("-" * 100)
    for r in records:
        bpb = f"{r['final_bpb']:.4f}" if r['final_bpb'] else "N/A"
        acc = f"{r['final_acc']:.4f}" if r['final_acc'] else "N/A"
        print(f"{r['phase']:>5} | {r['model']:>12} | {r['train_steps']:>7} | {r['effective_batch']:>5} | {r['seq_len']:>4} | {r['params']:>7} | {r['wall_time_h']:>7.1f} | {r['throughput_bytes_s']:>7} | {bpb:>6} | {acc:>6}")
    print(f"\n{len(records)} experiments recorded → {TRACKER_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("record")
    p.add_argument("--exp", required=True)
    p.add_argument("--phase", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--train-corpus", default="corpus_v3.txt")
    p.add_argument("--eval-corpus", default="corpus_v3.txt")
    p.add_argument("--train-steps", type=int, default=50000)
    p.add_argument("--effective-batch", type=int, default=0)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--params", default="?M")
    p.add_argument("--wall-time", type=float, default=0)
    p.add_argument("--throughput", type=int, default=0)
    p.add_argument("--bpb", type=float, default=None)
    p.add_argument("--acc", type=float, default=None)
    p.add_argument("--notes", default="")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("table")
    p.set_defaults(func=cmd_table)

    args = parser.parse_args()
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
