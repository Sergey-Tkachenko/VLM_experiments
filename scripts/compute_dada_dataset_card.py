#!/usr/bin/env python3
"""
Compute summary statistics + a Markdown dataset card for DADA-2000 annotations.

Default input is the XLSX you shared:
  /mnt/data/dada_text_annotations.xlsx

Usage:
  python compute_dada_dataset_card.py \
    --xlsx /path/to/dada_text_annotations.xlsx \
    --out_md dada_dataset_card.md \
    --fps 30 \
    --resolution "1584x660"
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def summarize(series: pd.Series | np.ndarray) -> dict:
    s = pd.Series(series).dropna().astype(float)
    if len(s) == 0:
        return {}
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "min": float(s.min()),
        "max": float(s.max()),
        "q05": float(s.quantile(0.05)),
        "q25": float(s.quantile(0.25)),
        "q75": float(s.quantile(0.75)),
        "q95": float(s.quantile(0.95)),
    }


def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100 * x:.{digits}f}%"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def summary_block(name: str, s: dict, fps: float) -> str:
    if not s:
        return ""
    def f(v: float) -> str:
        return str(int(v)) if float(v).is_integer() else f"{v:.2f}"

    mean_s = s["mean"] / fps
    med_s = s["median"] / fps
    q25_s = s["q25"] / fps
    q75_s = s["q75"] / fps
    return (
        f"- **{name}** (frames): mean {f(s['mean'])}, median {f(s['median'])}, "
        f"p05 {f(s['q05'])}, p25 {f(s['q25'])}, p75 {f(s['q75'])}, p95 {f(s['q95'])} "
        f"(≈ seconds @ {fps:g}fps: mean {mean_s:.2f}s, median {med_s:.2f}s, "
        f"p25 {q25_s:.2f}s, p75 {q75_s:.2f}s)\n"
    )


def dist(df: pd.DataFrame, col: str, mapping: dict[int, str]) -> list[dict]:
    vc = df[col].value_counts(dropna=False).sort_index()
    total = int(vc.sum())
    out = []
    for code, count in vc.items():
        code_int = int(code)
        out.append(
            {
                "code": code_int,
                "name": mapping.get(code_int, str(code_int)),
                "count": int(count),
                "pct": float(count / total),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", type=Path, required=True, help="Path to dada_text_annotations.xlsx")
    ap.add_argument("--out_md", type=Path, required=True, help="Where to write the dataset card markdown")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--resolution", type=str, default="1584x660")
    args = ap.parse_args()

    df = pd.read_excel(args.xlsx, sheet_name="Sheet1")

    col_acc = "whether an accident occurred (1/0)"
    acc = df[col_acc].astype(int) == 1

    frames = df["total frames"].astype(int)
    t_ai = df["abnormal start frame"].astype(int)
    t_ae = df["abnormal end frame"].astype(int)
    t_co = df["accident frame"].astype(int)

    abnormal_dur = (t_ae - t_ai).astype(int)
    tail_dur = (frames - t_ae).astype(int)
    pre_abnormal = t_ai.astype(int)

    pre_collision = np.where(acc, (t_co - t_ai), np.nan)
    post_collision = np.where(acc, (t_ae - t_co), np.nan)

    ab_ratio = abnormal_dur / frames
    acc_pos = np.where(acc, t_co / frames, np.nan)

    # categorical mappings
    weather_map = {1: "sunny", 2: "rainy", 3: "snowy", 4: "foggy"}
    light_map = {1: "day", 2: "night"}
    scene_map = {1: "highway", 2: "tunnel", 3: "mountain", 4: "urban", 5: "rural"}
    linear_map = {1: "arterials", 2: "curve", 3: "intersection", 4: "T-junction", 5: "ramp"}

    weather_dist = dist(df, "weather(sunny,rainy,snowy,foggy)1-4", weather_map)
    light_dist = dist(df, "light(day,night)1-2", light_map)
    scene_dist = dist(df, "scenes(highway,tunnel,mountain,urban,rural)1-5", scene_map)
    linear_dist = dist(df, "linear(arterials,curve,intersection,T-junction,ramp) 1-5", linear_map)

    # type table
    type_ids = df["type"].astype(int)
    ct = pd.crosstab(type_ids, acc).reindex(columns=[False, True], fill_value=0)
    ct.columns = ["non_accident", "accident"]
    ct["total"] = ct["non_accident"] + ct["accident"]
    ct["accident_rate"] = ct["accident"] / ct["total"]
    ct = ct.sort_values("total", ascending=False)

    # text sheet sanity
    text_df = pd.read_excel(args.xlsx, sheet_name="text", header=None).dropna(how="all")
    ids = []
    current = None
    for v in text_df[0].tolist():
        if pd.notna(v):
            current = int(v)
        ids.append(current)
    text_df["group_id"] = ids

    # quality
    missing_texts = int(df["texts"].isna().sum())
    missing_measures = int(df["measures"].isna().sum())
    unique_texts = int(df["texts"].astype(str).str.strip().nunique())

    # markdown tables
    def dist_rows(dlist):
        return [[d["name"], str(d["count"]), fmt_pct(d["pct"])] for d in dlist]

    type_rows = []
    for type_id, r in ct.reset_index().iterrows():
        pass  # placeholder for mypy; will rebuild below

    ct2 = ct.copy()
    ct2["share"] = ct2["total"] / len(df)
    ct2 = ct2.reset_index().rename(columns={"type": "type_id"})
    type_rows = [
        [
            str(int(r["type_id"])),
            str(int(r["total"])),
            str(int(r["accident"])),
            str(int(r["non_accident"])),
            fmt_pct(float(r["accident_rate"]), 2),
            fmt_pct(float(r["share"]), 2),
        ]
        for _, r in ct2.iterrows()
    ]

    md_lines = []
    md_lines.append("# DADA-2000 — Dataset card (from provided XLSX annotations)\n")
    md_lines.append(f"> Generated from: `{args.xlsx.name}` (sheets `Sheet1` + `text`).\n")
    md_lines.append("## 0. Purpose\n")
    md_lines.append(
        "DADA-2000 is a dashcam accident dataset designed for **accident understanding/anticipation** "
        "and **driver attention modeling**. This card summarizes only what is measurable from the "
        "provided XLSX annotation file.\n"
    )
    md_lines.append("**References (add your preferred canonical links):**\n")
    md_lines.append("- Paper: arXiv:1904.12634\n")
    md_lines.append("- GitHub: https://github.com/JWFangit/LOTVS-DADA\n")

    md_lines.append("## 1. General stats\n")
    md_lines.append(f"- Assumed video spec: **{args.resolution} @ {args.fps:g} fps**\n")
    md_lines.append(f"- Samples in `Sheet1` (rows): **{len(df):,}**\n")
    md_lines.append(f"- Unique `video` IDs in this XLSX: **{df['video'].nunique():,}**\n")
    md_lines.append(f"- Accident present: **{int(acc.sum()):,} / {len(df):,} ({fmt_pct(float(acc.mean()),2)})**\n")
    md_lines.append(f"- Total frames (sum of `total frames`): **{int(frames.sum()):,}**\n")

    md_lines.append("### Clip length (`total frames`)\n")
    md_lines.append(summary_block("All samples", summarize(frames), args.fps))
    md_lines.append(summary_block("Accident samples only", summarize(frames[acc]), args.fps))
    md_lines.append(summary_block("No-accident samples only", summarize(frames[~acc]), args.fps))

    md_lines.append("### Temporal structure (derived from frame markers)\n")
    md_lines.append(summary_block("Pre-abnormal context (`t_ai`)", summarize(pre_abnormal), args.fps))
    md_lines.append(summary_block("Abnormal window length (`t_ae - t_ai`)", summarize(abnormal_dur), args.fps))
    md_lines.append(summary_block("Time-to-collision (`t_co - t_ai`, accident only)", summarize(pre_collision), args.fps))
    md_lines.append(summary_block("Post-collision (`t_ae - t_co`, accident only)", summarize(post_collision), args.fps))
    md_lines.append(summary_block("Tail after abnormal (`end - t_ae`)", summarize(tail_dur), args.fps))

    md_lines.append("Additional ratios (computed):\n")
    md_lines.append(f"- Mean abnormal-window fraction of the clip: **{float(np.mean(ab_ratio)):.3f}** (median {float(np.median(ab_ratio)):.3f})\n")
    md_lines.append(f"- Mean accident-frame position in the clip (accident samples): **{float(np.nanmean(acc_pos)):.3f}** (median {float(np.nanmedian(acc_pos)):.3f})\n")

    md_lines.append("## 2. Metadata distributions\n")
    md_lines.append("### Weather\n")
    md_lines.append(md_table(["weather", "count", "share"], dist_rows(weather_dist)) + "\n")
    md_lines.append("### Light\n")
    md_lines.append(md_table(["light", "count", "share"], dist_rows(light_dist)) + "\n")
    md_lines.append("### Scene\n")
    md_lines.append(md_table(["scene", "count", "share"], dist_rows(scene_dist)) + "\n")
    md_lines.append("### Road geometry (`linear`)\n")
    md_lines.append(md_table(["road_type", "count", "share"], dist_rows(linear_dist)) + "\n")

    md_lines.append("## 3. GT categories (`type`) + accident/non-accident breakdown\n")
    md_lines.append(f"- # unique `type` IDs: **{type_ids.nunique()}**\n")
    md_lines.append(md_table(["type_id", "total", "accident", "non_accident", "accident_rate", "share"], type_rows) + "\n")

    md_lines.append("## 4. Text fields / quality notes\n")
    md_lines.append(f"- `texts` missing: {missing_texts}; `measures` missing: {missing_measures}\n")
    md_lines.append(f"- Unique normalized `texts` strings (after `.strip()`): {unique_texts}\n")
    md_lines.append("- Many `texts` values include trailing spaces — strip before using.\n")
    md_lines.append(f"- `text` sheet groups present: {text_df['group_id'].nunique()} (rows: {len(text_df)})\n")

    args.out_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Wrote dataset card: {args.out_md}")


if __name__ == "__main__":
    main()
