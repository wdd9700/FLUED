"""Render v3.4 boundary ROI JSON as a self-contained text/heatmap HTML."""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path


def _confidence_background(value: float) -> str:
    value = max(-1.0, min(1.0, float(value)))
    if value < 0:
        alpha = 0.12 + 0.38 * abs(value)
        return f"rgba(43, 120, 197, {alpha:.3f})"
    alpha = 0.12 + 0.38 * value
    return f"rgba(208, 74, 58, {alpha:.3f})"


def _display_char(value: str) -> str:
    if value == " ":
        return "·"
    if value == "\n":
        return "↵"
    if value == "\r":
        return "␍"
    if value == "\t":
        return "⇥"
    return value


def _cell(byte: dict) -> str:
    classes = ["byte"]
    if byte["model_hard_boundary"]:
        classes.append("model")
    if byte["logic_transition"]:
        classes.append("logic")
    if byte["utf8_continuation"]:
        classes.append("utf8")
    if byte["forced_max_span_boundary"]:
        classes.append("forced")
    markers = []
    if byte["model_hard_boundary"]:
        markers.append("模型边界")
    if byte["logic_transition"]:
        markers.append("逻辑转折")
    if byte["utf8_continuation"]:
        markers.append("UTF-8 continuation")
    if byte["forced_max_span_boundary"]:
        markers.append("强制 max-span")
    title = (
        f"byte {byte['index']} {byte['hex']} | char={byte['char']!r} | "
        f"confidence={byte['signed_confidence']:.4f} | chunk={byte['chunk_id']} "
        f"offset={byte['chunk_offset']} | " + (", ".join(markers) if markers else "普通字节")
    )
    char = _display_char(byte["char"])
    return (
        f'<span class="{" ".join(classes)}" style="background:{_confidence_background(byte["signed_confidence"])}" '
        f'title="{escape(title)}"><b>{escape(char)}</b><small>{byte["index"]:03d}</small>'
        f'<em>{escape(byte["hex"])}</em></span>'
    )


def render(payload: dict) -> str:
    checkpoint = payload.get("checkpoint", {})
    checkpoint_text = checkpoint.get("checkpoint") or "随机初始化（smoke）"
    sections = []
    for case in payload.get("cases", []):
        summary = case["summary"]
        budget = case["budget"]
        tags = " ".join(f"<span class=tag>{escape(str(tag))}</span>" for tag in case.get("tags", []))
        targets = "；".join(escape(str(item)) for item in case.get("audit_targets", []))
        pair_meta = (
            f"category={escape(str(case.get('category', 'uncategorized')))} · "
            f"pair_id={escape(str(case.get('pair_id', '-')))} · "
            f"variant={escape(str(case.get('variant', '-')))}"
        )
        cells = "".join(_cell(byte) for byte in case["bytes"])
        sections.append(
            f"""<section class=case>
<div class=case-head><div><h2>{escape(case['title'])}</h2><div>{tags}</div></div>
<div class=metrics>bytes {case['byte_length']} · chunks {summary['active_chunk_count']} · 
hard boundaries {summary['hard_chunk_boundary_count']} · forced max-span {summary['forced_max_span_boundary_count']} · 
readout hard {summary['active_readout_slots_hard']} / soft {summary['active_readout_slots_soft']:.2f}</div></div>
<div class=pair-meta>{pair_meta}</div>
<p class=targets><b>人工审阅目标：</b>{targets}</p>
<pre class=text>{escape(case['text'])}</pre>
<div class=budget>budget: max_chunks={budget['max_chunks']}, max_span={budget['max_span']}, 
bytes_per_chunk={budget['bytes_per_chunk_budget']}, readout_vectors={budget['configured_readout_vectors']}</div>
<div class=heatmap>{cells}</div>
<div class=chunk-list>{''.join(f"<span>chunk {c['chunk_id']}: {c['byte_length']}B, readout hard={c['readout_slots_hard']}, soft={c['readout_slots_soft']:.2f}</span>" for c in case['chunks'])}</div>
</section>"""
        )
    return f"""<!doctype html>
<html lang=zh-CN><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>FLUED v3.4 Boundary ROI</title>
<style>
:root {{ color-scheme: light; font-family: Inter, "Segoe UI", "Noto Sans CJK SC", sans-serif; color:#20252b; background:#f4f6f8; }}
body {{ max-width:1500px; margin:0 auto; padding:28px; }}
h1 {{ margin:0 0 8px; font-size:26px; }} h2 {{ margin:0 0 8px; font-size:19px; }}
.sub {{ color:#5d6872; margin:0 0 18px; }}
.legend, .case {{ background:#fff; border:1px solid #dbe1e6; border-radius:6px; }}
.legend {{ padding:14px 16px; margin:14px 0 22px; display:flex; flex-wrap:wrap; gap:12px 18px; align-items:center; }}
.key {{ display:inline-flex; gap:6px; align-items:center; font-size:13px; }}
.swatch {{ width:18px; height:12px; display:inline-block; border:1px solid #9aa5af; }}
.swatch.model {{ border-top:4px solid #d04a3a; }} .swatch.logic {{ border-right:4px solid #d99a27; }}
.swatch.utf8 {{ border-bottom:4px solid #2b78c5; }} .swatch.forced {{ border-left:4px solid #7a58b8; }}
.case {{ padding:18px; margin:0 0 20px; overflow:hidden; }}
.case-head {{ display:flex; gap:18px; justify-content:space-between; align-items:flex-start; }}
.metrics {{ color:#45525d; font-size:13px; text-align:right; max-width:55%; }}
.tag {{ display:inline-block; background:#eef2f5; color:#4b5964; padding:3px 7px; margin:0 5px 4px 0; border-radius:3px; font-size:12px; }}
.targets {{ color:#4d5962; font-size:13px; margin:10px 0; }}
.text {{ white-space:pre-wrap; overflow-wrap:anywhere; background:#fafbfc; border-left:3px solid #b9c4cc; padding:12px; margin:12px 0; line-height:1.55; }}
.budget {{ color:#586671; font:12px ui-monospace, SFMono-Regular, Consolas, monospace; margin:8px 0; }}
.heatmap {{ display:flex; flex-wrap:wrap; gap:3px; padding:10px; background:#f7f9fa; border:1px solid #e0e5e9; max-height:330px; overflow:auto; }}
.byte {{ position:relative; display:inline-flex; flex-direction:column; width:42px; height:45px; justify-content:center; align-items:center; border:1px solid #d6dde2; border-radius:2px; font:13px ui-monospace, SFMono-Regular, Consolas, monospace; }}
.byte b {{ font-size:15px; line-height:17px; max-width:39px; overflow:hidden; }} .byte small {{ color:#53616c; font-size:9px; line-height:10px; }}
.byte em {{ color:#64717b; font-size:9px; line-height:10px; font-style:normal; }}
.byte.model {{ border-top:4px solid #d04a3a; }} .byte.logic {{ border-right:4px solid #d99a27; }}
.byte.utf8 {{ box-shadow:inset 0 -4px 0 #2b78c5; }} .byte.forced {{ border-left:4px solid #7a58b8; }}
.chunk-list {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }}
.chunk-list span {{ background:#f0f3f5; padding:4px 7px; border-radius:3px; font:11px ui-monospace, SFMono-Regular, Consolas, monospace; }}
@media (max-width:700px) {{ body {{ padding:14px; }} .case-head {{ display:block; }} .metrics {{ text-align:left; max-width:none; margin-top:10px; }} }}
</style></head><body>
<h1>FLUED v3.4 边界 ROI / 切分行为审阅</h1>
<p class=sub>设备：{escape(str(payload.get('device', 'unknown')))} · checkpoint：{escape(str(checkpoint_text))} · 
样本数：{len(payload.get('cases', []))} · 仅推理，不包含训练</p>
<div class=legend>
<span class=key><i class="swatch model"></i>红色上边：模型 hard boundary</span>
<span class=key><i class="swatch logic"></i>黄色右边：逻辑转折 / soft transition</span>
<span class=key><i class="swatch utf8"></i>蓝色下边：UTF-8 continuation</span>
<span class=key><i class="swatch forced"></i>紫色左边：强制 max-span boundary</span>
<span class=key>底色：signed confidence，蓝=continuation pressure，红=cut pressure</span>
</div>
{''.join(sections)}
</body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json")
    parser.add_argument("--output-dir", default="outputs/v34_boundary_roi")
    parser.add_argument("--output-name", default="v34_boundary_roi.html")
    args = parser.parse_args()
    payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / args.output_name
    destination.write_text(render(payload), encoding="utf-8")
    print(json.dumps({"output": str(destination), "case_count": len(payload.get("cases", []))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
