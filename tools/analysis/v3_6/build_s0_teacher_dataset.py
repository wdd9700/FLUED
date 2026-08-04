"""Build the S0 teacher-labeled boundary dataset via a local LM Studio teacher.

Pipeline: parse the user's hand-annotated samples -> few-shot prompt -> local
OpenAI-compatible endpoint inserts '|' markers into fresh corpus samples ->
strict validation (output must equal input with only '|' inserted) ->
convert marker positions to UTF-8 byte offsets -> JSONL + spot-check file.

The teacher never counts offsets; marker insertion only. Validation rejects
any output that alters, adds, or drops a single character.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

RULES = """你是语义切分标注员。唯一任务：在文本中插入半角竖线 | 标记语义段边界。
铁律（违反任何一条即作废）：
- 只插入 |，绝对不改、不增、不删任何原有字符（包括空白、换行、全角符号、代码缩进）；
- 输出只有标注后的文本本身，不要解释、前言、代码块包裹或复述。
切分规则（从人工范本总结）：
1. 语义单元必须完整：实体、专名、引号内容、括号注释、LaTeX 公式、日期表达式、版本号、URL、下划线空白，一律不劈开；
2. 数字和单位/量词在一起（如 701人、70.2%、24%、1 January 2006）；
3. 标点挂在它所属段的段尾；
4. 话语标记、连接词、功能词可以独立成段（据悉/此外/就是/是由/via/so that/Refusing to）；
5. 枚举项各自成段（产、学、研/线上/线下/一、/1、/-）；
6. 年份作专名修饰时切开（2020|上海广告标识技术展览会），完整日期则整体成段；
7. 粒度参考范本：中位约 7 个汉字或 3-5 个英文单词一段。单段超过 12 个汉字或 6 个英文单词时通常应再切；整句不切是错误；拿不准就少切，不许硬凑；
8. 代码与对话标记（```、函数签名、[Human]:、<eoh> 等）保持原样：在语句/逻辑行之间切，绝不在一行代码内部切。"""


_QUOTE_RULES = RULES


def _rules() -> str:
    return _QUOTE_RULES


def parse_exemplars(path: Path, max_exemplars: int) -> list[tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"### 样本\d+（语料 [\d]+% 处）\n", text)[1:]
    exemplars = []
    for block in blocks:
        annotated = "\n".join(l for l in block.strip().split("\n") if l.strip())
        if not annotated:
            continue
        raw = annotated.replace("|", "")
        exemplars.append((raw, annotated))
    return exemplars[:max_exemplars]


def build_prompt(exemplars: list[tuple[str, str]], target: str) -> list[dict]:
    shots = []
    for i, (raw, annotated) in enumerate(exemplars, 1):
        shots.append(f"【范本{i}输入】\n{raw}\n【范本{i}输出】\n{annotated}")
    system = _rules() + "\n\n以下是人工标注范本，严格模仿其切分风格与粒度：\n\n" + "\n\n".join(shots)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"【待标注输入】\n{target}\n【待标注输出】\n"},
    ]


def call_teacher(endpoint: str, model: str, messages: list[dict], timeout: int, api_key: str = "") -> str:
    body: dict = {"model": model, "messages": messages, "max_tokens": 16384}
    if os.environ.get("S0_TEACHER_TEMPERATURE"):
        body["temperature"] = float(os.environ["S0_TEACHER_TEMPERATURE"])
    if os.environ.get("S0_TEACHER_NO_THINKING"):
        body["thinking"] = {"type": "disabled"}
    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/chat/completions",
        data=payload,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


QUOTE_CANON = {"“": '"', "”": '"', "‘": "'", "’": "'"}


def validate_and_convert(target: str, output: str) -> list[int] | None:
    cleaned = output.strip()
    bare = []
    boundaries_chars = []
    for ch in cleaned:
        if ch == "|":
            boundaries_chars.append(len(bare))
        else:
            bare.append(QUOTE_CANON.get(ch, ch))
    canon_target = [QUOTE_CANON.get(ch, ch) for ch in target]
    if bare != canon_target:
        return None
    boundaries = []
    for char_idx in boundaries_chars:
        boundaries.append(len(target[:char_idx].encode("utf-8")))
    return boundaries


JSON_INSTR = """输出格式：只输出一个 JSON 数组，每个元素是一个语义段的原文片段，顺序拼接必须逐字符等于输入原文。
例：["第一段原文", "第二段原文", "第三段原文"]
不要输出任何其他内容（不要解释、不要 markdown 代码块、不要键名）。段内空白和换行必须原样保留。"""


def build_prompt_json(exemplars: list[tuple[str, str]], target: str) -> list[dict]:
    shots = []
    for i, (raw, annotated) in enumerate(exemplars, 1):
        segs = [s for s in annotated.split("|") if s]
        shots.append(f"【范本{i}输入】\n{raw}\n【范本{i}输出】\n{json.dumps(segs, ensure_ascii=False)}")
    system = _rules() + "\n\n" + JSON_INSTR + "\n\n以下是人工标注范本，严格模仿其切分风格与粒度：\n\n" + "\n\n".join(shots)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"【待标注输入】\n{target}\n【待标注输出】\n"},
    ]


def _extract_json_array(text: str) -> list[str] | None:
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return None
    return data


def validate_json_mode(target: str, output: str) -> list[int] | None:
    segs = _extract_json_array(output.strip())
    if segs is None:
        return None
    joined = "".join(segs)
    canon = lambda s: [QUOTE_CANON.get(ch, ch) for ch in s]
    if canon(joined) != canon(target):
        return None
    boundaries = []
    offset = 0
    for seg in segs[:-1]:
        offset += len(seg.encode("utf-8"))
        boundaries.append(offset)
    return boundaries


def sample_candidates(corpus: Path, count: int, seed: int) -> list[dict]:
    # corpus may be a single file or a directory of text shards
    if corpus.is_dir():
        files = sorted(corpus.glob("*.txt"))
    else:
        files = [corpus]
    sizes = [(f, f.stat().st_size) for f in files]
    rng = random.Random(seed)
    rows = []
    while len(rows) < count:
        fh_path, size = rng.choice(sizes)
        frac = rng.uniform(0.0, 0.94)
        with fh_path.open("rb") as fh:
            fh.seek(int(size * frac))
            block = fh.read(65536).decode("utf-8", errors="ignore")
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if len(lines) < 3:
            continue
        text = "\n".join(lines[1:4])
        if 120 <= len(text) <= 600 and not text.startswith("{") and "�" not in text:
            rows.append({"offset_frac": round(frac, 6), "shard": fh_path.name, "text": text})
    return rows


def label_one(
    index: int,
    cand: dict,
    exemplars: list[tuple[str, str]],
    endpoint: str,
    model: str,
    timeout: int,
    output_mode: str = "marks",
    api_key: str = "",
    attempts: int = 2,
) -> tuple[dict | None, dict | None]:
    target = cand["text"]
    boundaries = None
    last_output = ""
    prompt_fn = build_prompt_json if output_mode == "json" else build_prompt
    validate_fn = validate_json_mode if output_mode == "json" else validate_and_convert
    for _attempt in range(attempts):
        try:
            last_output = call_teacher(endpoint, model, prompt_fn(exemplars, target), timeout, api_key)
        except Exception as exc:
            last_output = f"__error__ {exc}"
        boundaries = validate_fn(target, last_output)
        if boundaries is not None:
            break
    if boundaries is None:
        return None, {"index": index, "text": target, "last_output": last_output[:500]}
    seg_bytes = []
    prev = 0
    for b in boundaries + [len(target.encode("utf-8"))]:
        seg_bytes.append(b - prev)
        prev = b
    record = {
        "index": index,
        "retry_of": cand.get("retry_index"),
        "offset_frac": cand["offset_frac"],
        "shard": cand.get("shard", ""),
        "text": target,
        "boundaries_bytes": boundaries,
        "n_segments": len(seg_bytes),
        "seg_bytes_median": sorted(seg_bytes)[len(seg_bytes) // 2],
        "annotated_text": last_output.strip(),
    }
    return record, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-file", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--endpoint", default="http://localhost:1234/v1")
    parser.add_argument("--api-key-file", default="", help="optional Bearer token file (e.g. Moonshot API key); never committed")
    parser.add_argument("--rules-file", default="", help="optional markdown rules file replacing the inline RULES (e.g. S05_TEACHER_RULES_CN.md)")
    parser.add_argument("--retry-file", default="", help="failures JSONL from a previous run; re-label those texts instead of sampling the corpus")
    parser.add_argument("--attempts", type=int, default=2, help="label attempts per sample before giving up")
    parser.add_argument("--model", required=True)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--max-exemplars", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-mode", choices=["marks", "json"], default="marks")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    exemplars = parse_exemplars(Path(cli.samples_file), cli.max_exemplars)
    print(f"[s0] exemplars={len(exemplars)}")
    if cli.rules_file:
        global _QUOTE_RULES
        _QUOTE_RULES = Path(cli.rules_file).read_text(encoding="utf-8")
        print(f"[s0] rules file: {cli.rules_file} ({len(_QUOTE_RULES)} chars)")
    if cli.dry_run:
        prompt = build_prompt(exemplars, "<示例目标文本>")
        print(prompt[0]["content"][:2000])
        print(f"[dry-run] user prompt chars={len(prompt[1]['content'])}")
        return

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    api_key = Path(cli.api_key_file).read_text(encoding="utf-8").strip() if cli.api_key_file else ""
    if cli.retry_file:
        candidates = [
            {"offset_frac": 0.0, "shard": "", "text": r["text"], "retry_index": r.get("index")}
            for r in (json.loads(l) for l in Path(cli.retry_file).read_text(encoding="utf-8").splitlines() if l.strip())
        ]
        print(f"[s0] retry candidates={len(candidates)} from {cli.retry_file}")
    else:
        candidates = sample_candidates(Path(cli.corpus), cli.count, cli.seed)
    ds_path = out_dir / "s0_teacher_labels.jsonl"
    spot_path = out_dir / "s0_spot_check.txt"
    fail_path = out_dir / "s0_failures.jsonl"
    rng = random.Random(cli.seed + 1)
    kept = failed = done = 0
    spot_rows = []
    t0 = time.time()
    with ds_path.open("w", encoding="utf-8") as sink, fail_path.open("w", encoding="utf-8") as fsink:
        with ThreadPoolExecutor(max_workers=cli.workers) as pool:
            futures = {
                pool.submit(label_one, i, cand, exemplars, cli.endpoint, cli.model, cli.timeout, cli.output_mode, api_key, cli.attempts): i
                for i, cand in enumerate(candidates)
            }
            for future in as_completed(futures):
                record, failure = future.result()
                done += 1
                if record is None:
                    failed += 1
                    fsink.write(json.dumps(failure, ensure_ascii=False) + "\n")
                else:
                    kept += 1
                    record.pop("annotated_text") if rng.random() >= 0.08 else spot_rows.append(
                        f"### index {record['index']}（语料 {record['offset_frac']*100:.1f}% 处）\n{record.pop('annotated_text')}\n"
                    )
                    sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                if done % 50 == 0:
                    rate = done / max(time.time() - t0, 1e-9)
                    print(f"[s0] {done}/{len(candidates)} kept={kept} failed={failed} {rate:.1f}/s", flush=True)
    spot_path.write_text("\n".join(spot_rows), encoding="utf-8")
    stats = {"kept": kept, "failed": failed, "spot_check_rows": len(spot_rows), "elapsed_sec": time.time() - t0}
    (out_dir / "s0_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
