#!/usr/bin/env python3
"""Analyze SWE-bench task structure: how far is 'write one function' from
'fix a real repo issue'?

Reads SWE-bench Verified (500 tasks), SWE-bench full (2294 tasks) and
HumanEval (164 tasks), and reports:
  - files touched per gold patch
  - whether the fix touches a different file than the failing test lives in
  - how many FAIL_TO_PASS tests gate each fix
  - problem statement length
  - HumanEval comparison (1 function, test in prompt)

Data sources (parquet):
  - https://hf-mirror.com/datasets/princeton-nlp/SWE-bench_Verified
  - https://hf-mirror.com/datasets/princeton-nlp/SWE-bench
  - https://hf-mirror.com/datasets/openai/openai_humaneval
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
DATA = Path(__file__).resolve().parents[3] / ".research_tmp"  # workspace/.research_tmp
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def load(name: str) -> "pd.DataFrame":
    import pandas as pd
    path = {
        "verified": DATA / "swebench_verified.parquet",
        "full": DATA / "swebench_full_test.parquet",
        "humaneval": DATA / "humaneval.parquet",
    }[name]
    return pq.read_table(str(path)).to_pandas()


def files_in_diff(diff: str) -> list[str]:
    """Return file paths touched by a unified diff."""
    if not diff:
        return []
    return re.findall(r"^diff --git a/(.+?) b/", diff, re.M)


def patch_stats(diff: str) -> dict:
    """Count lines added/removed (approximation: skip ---/+++ headers)."""
    added = removed = 0
    for line in (diff or "").splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return {"added": added, "removed": removed, "total": added + removed}


def main() -> None:
    import pandas as pd

    results = {}

    # ---------- SWE-bench Verified ----------
    ver = load("verified")
    rows = []
    for _, r in ver.iterrows():
        patch_files = files_in_diff(r["patch"])
        test_files = files_in_diff(r["test_patch"])
        ps = patch_stats(r["patch"])
        ftp = json.loads(r["FAIL_TO_PASS"]) if r["FAIL_TO_PASS"] else []
        ptp = json.loads(r["PASS_TO_PASS"]) if r["PASS_TO_PASS"] else []
        rows.append({
            "repo": r["repo"],
            "patch_files": len(patch_files),
            "test_files": test_files,
            "added": ps["added"],
            "removed": ps["removed"],
            "total_lines": ps["total"],
            "n_ftp": len(ftp),
            "n_ptp": len(ptp),
            "n_tests_gate": len(ftp) + len(ptp),
            "stmt_len": len(r["problem_statement"] or ""),
            "has_hint": bool(r.get("hints_text")),
            "difficulty": r.get("difficulty"),
        })
    vdf = pd.DataFrame(rows)

    results["verified"] = {
        "n": len(vdf),
        "n_repos": vdf["repo"].nunique(),
        "repos": sorted(vdf["repo"].unique()),
        "files_per_patch_mean": round(vdf["patch_files"].mean(), 2),
        "files_per_patch_median": float(vdf["patch_files"].median()),
        "files_dist": vdf["patch_files"].value_counts().sort_index().to_dict(),
        "multi_file_share": round((vdf["patch_files"] >= 2).mean() * 100, 1),
        "single_file_share": round((vdf["patch_files"] == 1).mean() * 100, 1),
        "lines_total_mean": round(vdf["total_lines"].mean(), 1),
        "lines_total_median": float(vdf["total_lines"].median()),
        "added_mean": round(vdf["added"].mean(), 1),
        "n_ftp_mean": round(vdf["n_ftp"].mean(), 2),
        "n_ftp_median": float(vdf["n_ftp"].median()),
        "n_ptp_mean": round(vdf["n_ptp"].mean(), 1),
        "n_ptp_median": float(vdf["n_ptp"].median()),
        "n_gate_mean": round(vdf["n_tests_gate"].mean(), 1),
        "n_gate_median": float(vdf["n_tests_gate"].median()),
        "stmt_len_mean": round(vdf["stmt_len"].mean()),
        "stmt_len_median": float(vdf["stmt_len"].median()),
        "has_hint_share": round(vdf["has_hint"].mean() * 100, 1),
        "difficulty": vdf["difficulty"].value_counts().to_dict(),
        "repos_top": vdf["repo"].value_counts().head(6).to_dict(),
    }

    # ---------- SWE-bench full ----------
    full = load("full")
    frows = []
    for _, r in full.iterrows():
        patch_files = files_in_diff(r["patch"])
        ps = patch_stats(r["patch"])
        ftp = json.loads(r["FAIL_TO_PASS"]) if r["FAIL_TO_PASS"] else []
        ptp = json.loads(r["PASS_TO_PASS"]) if r["PASS_TO_PASS"] else []
        frows.append({
            "patch_files": len(patch_files),
            "total_lines": ps["total"],
            "n_ftp": len(ftp),
            "n_ptp": len(ptp),
            "n_tests_gate": len(ftp) + len(ptp),
            "stmt_len": len(r["problem_statement"] or ""),
        })
    fdf = pd.DataFrame(frows)
    results["full"] = {
        "n": len(fdf),
        "files_per_patch_mean": round(fdf["patch_files"].mean(), 2),
        "multi_file_share": round((fdf["patch_files"] >= 2).mean() * 100, 1),
        "n_gate_mean": round(fdf["n_tests_gate"].mean(), 1),
        "n_ptp_median": float(fdf["n_ptp"].median()),
        "stmt_len_mean": round(fdf["stmt_len"].mean()),
    }

    # ---------- HumanEval ----------
    he = load("humaneval")
    hstmt = [len(x or "") for x in he["prompt"]]
    htest = [len(x or "") for x in he["test"]]
    results["humaneval"] = {
        "n": len(he),
        "prompt_len_mean": round(sum(hstmt) / len(hstmt)),
        "test_len_mean": round(sum(htest) / len(htest)),
        "single_function": "one function per task, test included in prompt",
    }

    with open(OUT / "stats.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))

    # ---------- Figures ----------
    # Fig 1: files per patch, Verified vs Full
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.arange(0, max(vdf["patch_files"].max(), fdf["patch_files"].max()) + 2) - 0.5
    ax.hist([vdf["patch_files"], fdf["patch_files"]], bins=bins, label=["SWE-bench Verified (500)", "SWE-bench full (2,294)"], color=["#2f6fed", "#9db9ee"], alpha=0.9)
    ax.set_xlabel("Files touched by the gold fix")
    ax.set_ylabel("Tasks")
    ax.set_xticks(range(0, 11))
    ax.legend(frameon=False)
    ax.set_title("Files touched by the gold fix (Verified vs full)")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_files_per_patch.png")
    plt.close(fig)

    # Fig 2: two test bars - few FAIL_TO_PASS vs huge PASS_TO_PASS (Verified)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    ax = axes[0]
    bins = np.arange(0, vdf["n_ftp"].max() + 1.5, 1) - 0.5
    ax.hist(vdf["n_ftp"], bins=bins, color="#2f6fed")
    ax.axvline(vdf["n_ftp"].median(), color="#e06c00", ls="--", label=f"median {vdf['n_ftp'].median():.0f}")
    ax.set_xlabel("FAIL_TO_PASS (tests that must start passing)")
    ax.set_ylabel("Tasks")
    ax.set_xticks(range(0, 13))
    ax.legend(frameon=False)
    ax.set_title("Newly enabled tests: few")
    ax = axes[1]
    bins = np.arange(0, vdf["n_ptp"].max() + 40, 40)
    ax.hist(vdf["n_ptp"], bins=bins, color="#9db9ee")
    ax.axvline(vdf["n_ptp"].median(), color="#e06c00", ls="--", label=f"median {vdf['n_ptp'].median():.0f}")
    ax.set_xlabel("PASS_TO_PASS (tests that must keep passing)")
    ax.set_ylabel("Tasks")
    ax.legend(frameon=False)
    ax.set_title("Regression tests: many")
    fig.suptitle("SWE-bench Verified: two test bars a fix must clear", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_tests_gate.png", bbox_inches="tight")
    plt.close(fig)

    # Fig 3: statement/prompt length boxplots
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bp = ax.boxplot([vdf["stmt_len"], fdf["stmt_len"], hstmt], tick_labels=["SWE-bench\nVerified", "SWE-bench\nfull", "HumanEval"], showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#2f6fed", "#9db9ee", "#e0a458"]):
        patch.set_facecolor(c)
    ax.set_ylabel("Problem / prompt length (chars)")
    ax.set_title("Issue descriptions are longer and vaguer than coding prompts")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_context_len.png")
    plt.close(fig)

    # Fig 4: human difficulty rating of Verified tasks (hours a human needs)
    diff_order = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
    counts = [vdf["difficulty"].get(d, 0) for d in diff_order]
    labels = ["<15 min", "15 min–1 h", "1–4 h", ">4 h"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, counts, color=["#7fb069", "#2f6fed", "#e0a458", "#d1495b"])
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, str(v), ha="center")
    ax.set_ylabel("Tasks")
    ax.set_title("Human-rated difficulty of the 500 'Verified' issues")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_human_difficulty.png")
    plt.close(fig)

    # Fig 5: gold patch size (lines) - fix is tiny once located
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.arange(0, 60, 2)
    ax.hist(vdf["total_lines"], bins=bins, color="#2f6fed")
    ax.axvline(vdf["total_lines"].median(), color="#e06c00", ls="--", label=f"median {vdf['total_lines'].median():.0f} lines")
    ax.set_xlabel("Lines changed by the gold fix")
    ax.set_ylabel("Tasks")
    ax.legend(frameon=False)
    ax.set_title("The fix is tiny once you know where to look")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_patch_size.png")
    plt.close(fig)

    print("\nfigures written to", OUT)


import numpy as np  # noqa: E402


if __name__ == "__main__":
    main()
