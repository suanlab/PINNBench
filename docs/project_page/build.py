#!/usr/bin/env python3
"""Build the PINNBench project page into docs/project_page/site/.

Every number and paragraph on the page comes from an artifact, not from this file:

* TL;DR, abstract and lay summary: the OpenReview camera-ready texts in
  CAMERA_SUBMISSION.md, whose abstract was checked character by character
  against page 1 of the camera-ready PDF.
* Figures (i) and (ii): the camera-ready figure PDFs, converted to SVG.
* Figure (iii): drawn here from
  experiments/results/paper_c/control_statistics/results.json.
* BibTeX: built from ``citeAs`` in dataset_pinnbench/croissant.json.

The manuscript, poster and CAMERA_SUBMISSION.md are local-only (git-ignored),
so this script runs on the authors' machine; the generated site/ is committed.

Usage (from the repo root):
    .venv/bin/python docs/project_page/build.py
"""

from __future__ import annotations

import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "docs" / "project_page"
SITE = HERE / "site"
ASSETS = SITE / "assets"
CAMERA = ROOT / "docs" / "paper_neurips_camera"
POSTER = CAMERA / "poster"
CONTROLS = ROOT / "experiments" / "results" / "paper_c" / "control_statistics" / "results.json"

BLUE, ROSE, SLATE = "#2563EB", "#F43F5E", "#94A3B8"


def form_text(heading: str) -> str:
    """The code block under ``heading`` in CAMERA_SUBMISSION.md."""
    s = (ROOT / "CAMERA_SUBMISSION.md").read_text()
    i = s.index(heading)
    j = s.index("```", i) + 3
    return s[j:s.index("```", j)].strip()


def lay_summary() -> str:
    """The OpenReview lay summary, with the seed count stated as in the paper.

    The submitted text says every setup was "repeated ten times"; only the core
    conditions use 10 seeds (main.tex: secondary studies use 3-5).
    """
    text = form_text("## Lay Summary*")
    old = "13 equation setups, each repeated ten times,"
    assert old in text
    return text.replace(old, "13 equation setups, with the core experiments repeated ten times,")


def tex_to_html(text: str) -> str:
    """Render the few inline TeX forms the abstract uses, and set policy names as code."""
    text = html.escape(text, quote=False)

    def math(m: re.Match) -> str:
        t = m.group(1)
        t = t.replace(r"\times", "×").replace(" ", "")
        t = re.sub(r"([A-Za-z])_(\w)", r"<i>\1</i><sub>\2</sub>", t)
        t = re.sub(r"^([a-z])(?==|$)", r"<i>\1</i>", t)
        return t

    text = re.sub(r"\$([^$]+)\$", math, text)
    return re.sub(r"\b(hybrid_full|hybrid_no_gate|pinn_only)\b", r"<code>\1</code>", text)


def bibtex() -> str:
    cite = json.loads((ROOT / "dataset_pinnbench" / "croissant.json").read_text())["citeAs"]
    m = re.match(r"(.+?)\. (.+?)\. In (.+?), (\d{4})\.$", cite)
    authors, title, venue, year = m.groups()
    names = [a.strip() for a in re.split(r",\s*(?:and\s+)?|\s+and\s+", authors) if a.strip()]
    bib_authors = " and ".join(f"{n.split()[-1]}, {' '.join(n.split()[:-1])}" for n in names)
    return (f"@inproceedings{{lee{year}pinnbench,\n"
            f"  title     = {{{title}}},\n"
            f"  author    = {{{bib_authors}}},\n"
            f"  booktitle = {{{venue}}},\n"
            f"  year      = {{{year}}}\n}}")


def controls_figure(out: Path) -> dict[str, float]:
    """Grouped bars of the epoch-matched controls; returns the means quoted on the page."""
    rows = json.loads(CONTROLS.read_text())
    label = {"heat1d": "Heat1D", "advdiff1d": "AdvDiff1D", "burgers1d": "Burgers (ν=0.01)",
             "burgers1d_verylow": "Burgers (ν=1e-4)", "kdv1d": "KdV1D",
             "allencahn1d": "AllenCahn (ε=0.01)", "allencahn1d_sharp": "AllenCahn (ε=0.001)",
             "reaction_diffusion1d_stiff": "ReactDiff (stiff)", "navier_stokes2d": "NavierStokes2D"}
    series = [("extra_training", "with pretraining",
               "extra Stage-3 training, uniform weights (pretrained)", BLUE),
              ("reweighting", "with pretraining", "causal reweighting (pretrained)", SLATE),
              ("reweighting", "without pretraining", "causal reweighting (no pretraining)", ROSE)]
    by = {(r["pde"], r["family"], r["comparison"]): r for r in rows}
    pdes = list(label)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "path"})
    fig, ax = plt.subplots(figsize=(6.0, 5.4))
    h = 0.26
    means = {}
    for k, (fam, comp, name, color) in enumerate(series):
        vals = [by[(p, fam, comp)]["delta_pct"] for p in pdes]
        sig = [by[(p, fam, comp)]["bh_fdr_significant"] for p in pdes]
        ys = [i + (k - 1) * h for i in range(len(pdes))]
        ax.barh(ys, vals, height=h, color=color, label=name, zorder=2)
        for y, v, s in zip(ys, vals, sig, strict=True):
            if s:
                ax.text(v + (0.25 if v >= 0 else -0.25), y, "*", va="center",
                        ha="left" if v >= 0 else "right", fontsize=11, color="#0F172A")
        means[f"{fam}/{comp}"] = sum(vals) / len(vals)
    ax.set_yticks(range(len(pdes)), [label[p] for p in pdes])
    ax.invert_yaxis()
    ax.axvline(0, color="#0F172A", lw=0.8, zorder=3)
    ax.set_xlim(-16, 11)
    ax.set_xlabel("Δ PDE residual (%), positive = lower residual")
    ax.grid(axis="x", color="#E2E8F0", zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, frameon=False, fontsize=11)
    fig.text(0.99, 0.01, "* BH-FDR significant (54 tests), 10 seeds", ha="right", fontsize=10,
             color="#475569")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return means


def pdf_to_svg(src: Path, dst: Path) -> None:
    subprocess.run(["pdftocairo", "-svg", str(src), str(dst)], check=True)


def main() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    ASSETS.mkdir(parents=True)

    shutil.copy(CAMERA / "main.pdf", ASSETS / "PINNBench_NeurIPS2026.pdf")
    shutil.copy(POSTER / "PINNBench_poster.pdf", ASSETS / "PINNBench_poster.pdf")
    for name in ("lab_logo.png", "smu_logo.png"):
        shutil.copy(POSTER / name, ASSETS / name)
    poster = Image.open(POSTER / "upload" / "PINNBench_poster_16x9.png").convert("RGB")
    poster.resize((2560, 1440), Image.LANCZOS).save(ASSETS / "poster.jpg", quality=88)
    poster.resize((1200, 675), Image.LANCZOS).save(ASSETS / "og.png", optimize=True)
    pdf_to_svg(CAMERA / "figures" / "fig_no_dominance.pdf", ASSETS / "fig_no_dominance.svg")
    pdf_to_svg(CAMERA / "figures" / "fig_dissociation.pdf", ASSETS / "fig_dissociation.svg")
    means = controls_figure(ASSETS / "fig_controls.svg")

    page = (HERE / "template.html").read_text()
    fills = {
        "TLDR": html.escape(form_text("## TL;DR"), quote=False),
        "LAY": html.escape(lay_summary(), quote=False),
        "ABSTRACT": tex_to_html(form_text("## Abstract*")),
        "BIBTEX": html.escape(bibtex(), quote=False),
        "UNIFORM": f"{means['extra_training/with pretraining']:+.1f}%",
        "RW_PRE": f"{means['reweighting/with pretraining']:+.1f}%",
        "RW_NOPRE": f"{means['reweighting/without pretraining']:+.1f}%",
    }
    for key, val in fills.items():
        if key.startswith(("UNI", "RW")):
            val = val.replace("-", "−")
        page = page.replace("{{" + key + "}}", val)
    assert "{{" not in page, re.findall(r"\{\{\w+\}\}", page)
    (SITE / "index.html").write_text(page)
    (SITE / "favicon.svg").write_text((HERE / "favicon.svg").read_text())
    (SITE / ".nojekyll").write_text("")
    print("built", SITE, {k: v for k, v in fills.items() if k.startswith(("UNI", "RW"))})


if __name__ == "__main__":
    main()
