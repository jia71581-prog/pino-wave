# DCLP-NO IEEE TGRS Paper and Figure Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert sealed DCLP-NO experiments into a visually polished, citation-verified, compiled IEEE TGRS manuscript with a graphical abstract, seven main figures, supplement, cover letter, and reproducibility manifest.

**Architecture:** Every number and plot is generated from immutable machine-readable artifacts. Deterministic vector scripts build diagrams and statistical plots; raster wavefields are exported at submission resolution. A claim audit reads `claim_gate.json` and selects the permitted title and wording before LaTeX compilation.

**Tech Stack:** IEEEtran LaTeX, BibTeX, Python 3.13, Matplotlib, NumPy, pandas, PyYAML, pdfinfo, pdffonts, Ghostscript, pytest.

---

## File Map

- Create `paper/tgrs_dclp_no/main.tex`: IEEEtran root.
- Create `paper/__init__.py`, `paper/tgrs_dclp_no/__init__.py`, and
  `paper/tgrs_dclp_no/figures/__init__.py`: make the deterministic figure style
  importable by tests and scripts.
- Create `paper/tgrs_dclp_no/references.bib`: verified primary sources.
- Create eight section files under `paper/tgrs_dclp_no/sections/`.
- Create `paper/tgrs_dclp_no/supplement/supplement.tex`.
- Create `paper/tgrs_dclp_no/cover_letter/cover_letter.tex`.
- Create `paper/tgrs_dclp_no/README.md`: exact build and artifact provenance.
- Create `paper/tgrs_dclp_no/figures/style.py`: shared Okabe-Ito style.
- Create figure scripts `scripts/make_tgrs_graphical_abstract.py` and
  `scripts/make_tgrs_fig1.py` through `scripts/make_tgrs_fig7.py`.
- Create `scripts/make_tgrs_tables.py`: generated LaTeX tables.
- Create `scripts/build_tgrs_paper.py`: claim-aware generation and compile.
- Create `scripts/verify_tgrs_submission.py`: final package audit.
- Create tests under `tests/tgrs_dclp_no/`.

### Task 1: Acquire and freeze the current IEEE template and author rules

**Files:**
- Create: `paper/tgrs_dclp_no/author_guidance_manifest.json`
- Create: `paper/tgrs_dclp_no/ieee_template/`

- [ ] **Step 1: Verify official sources**

At execution time, use the current official IEEE and GRSS pages only:

- `https://www.grss-ieee.org/publications/author-resources/tgrs-information-for-authors/`
- `https://www.grss-ieee.org/publications/transactions-on-geoscience-remote-sensing/`
- `https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/`
- `https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/prepare-supplementary-materials/`
- `https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/create-graphics-for-your-article/`

Record access date, final URL, page title, and SHA-256 of every downloaded
template file.

- [ ] **Step 2: Freeze the author-guidance manifest**

Write:

```json
{
  "schema": "dclp_no_tgrs_author_guidance_v1",
  "access_date": "2026-07-24",
  "journal": "IEEE Transactions on Geoscience and Remote Sensing",
  "template_class": "IEEEtran",
  "graphical_abstract": {
    "reviewed_with_article": true,
    "target_width_px": 660,
    "target_height_px": 295,
    "minimum_dpi": 300
  },
  "main_page_target": 10,
  "sources": []
}
```

Populate `sources` only with verified official records. If current rules differ
from these values, update the manifest and the build checks to the official
values before writing the manuscript.

### Task 2: Build a verified primary-source bibliography

**Files:**
- Create: `paper/tgrs_dclp_no/references.bib`
- Create: `paper/tgrs_dclp_no/reference_audit.csv`

- [ ] **Step 1: Search and verify the literature**

At execution time, use the academic research and citation workflow. Verify
primary sources for:

- Fourier Neural Operator;
- factorized Fourier neural operators;
- DeepONet and multi-input operator networks;
- neural operators for acoustic or seismic wave propagation;
- numerical dispersion in finite-difference wave solvers;
- the LWC-84 or exact optimized stencil lineage used by the code;
- Eikonal travel-time calculation;
- LoRA and parameter-efficient adaptation;
- phase, spectrum, and receiver-domain seismic evaluation.

For every reference, verify title, authors, venue, year, volume/pages when
applicable, DOI, and publisher or primary-preprint URL.

- [ ] **Step 2: Create the audit table**

Use columns:

```text
citation_key,title,year,doi,primary_url,verified_against,verification_date,used_in_section
```

Reject any row lacking a resolvable title and primary URL. Search-result snippets
are not acceptable evidence.

- [ ] **Step 3: Validate BibTeX**

Run:

```bash
/home/jiayh/miniconda3/bin/python - <<'PY'
from pathlib import Path
import re
text = Path("paper/tgrs_dclp_no/references.bib").read_text()
keys = re.findall(r"@[A-Za-z]+\\{([^,]+),", text)
assert keys and len(keys) == len(set(keys))
assert "doi" in text.lower()
print({"entries": len(keys), "unique": len(set(keys))})
PY
```

Expected: nonzero equal entry and unique counts.

### Task 3: Create a single publication style and figure provenance contract

**Files:**
- Create: `paper/__init__.py`
- Create: `paper/tgrs_dclp_no/__init__.py`
- Create: `paper/tgrs_dclp_no/figures/__init__.py`
- Create: `paper/tgrs_dclp_no/figures/style.py`
- Create: `tests/tgrs_dclp_no/test_figure_style.py`

- [ ] **Step 1: Write the style test**

```python
from paper.tgrs_dclp_no.figures.style import COLORS, apply_style


def test_style_has_okabe_ito_semantics():
    assert COLORS["truth"] == "#000000"
    assert COLORS["baseline"] == "#D55E00"
    assert COLORS["parent"] == "#0072B2"
    assert COLORS["lora"] == "#009E73"
    apply_style()
```

- [ ] **Step 2: Implement the style**

```python
from __future__ import annotations

import matplotlib as mpl

COLORS = {
    "truth": "#000000",
    "baseline": "#D55E00",
    "parent": "#0072B2",
    "lora": "#009E73",
    "observed": "#CC79A7",
    "future": "#56B4E9",
    "neutral": "#7A7A7A",
    "warning": "#E69F00",
}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.dpi": 600,
        }
    )


def save_vector_and_raster(fig, output_base) -> None:
    fig.savefig(f"{output_base}.pdf")
    fig.savefig(f"{output_base}.png", dpi=600)
```

- [ ] **Step 3: Run the test**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_figure_style.py
```

Expected: test passes.

### Task 4: Draw the graphical abstract and algorithm/network architecture

**Files:**
- Create: `scripts/make_tgrs_graphical_abstract.py`
- Create: `scripts/make_tgrs_fig1.py`
- Generate: `paper/tgrs_dclp_no/figures/graphical_abstract.*`
- Generate: `paper/tgrs_dclp_no/figures/fig1_architecture.*`

- [ ] **Step 1: Use the scientific-schematics workflow**

At execution time, invoke the scientific-schematics skill for layout review.
Use it to decide hierarchy and spacing; keep all final labels deterministic in
the vector script. Do not accept generated text embedded in a raster diagram.

- [ ] **Step 2: Implement reusable diagram primitives**

Both scripts define:

```python
def rounded_box(axis, xy, width, height, text, *, facecolor, edgecolor, fontsize=7):
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.025",
        linewidth=1.1,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    axis.add_patch(patch)
    axis.text(xy[0] + width / 2, xy[1] + height / 2, text,
              ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(axis, start, end, *, color="#425466", style="-|>", linewidth=1.2):
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": style, "color": color, "lw": linewidth},
    )
```

- [ ] **Step 3: Draw Fig. 1 at two-column width**

Use a 7.16 inch by 4.6 inch canvas with four horizontal stages:

1. `Velocity + source + query time`;
2. `Complex-FNO medium pyramid` and `Source encoder`;
3. `Travel time → τ, causal, sin/cos, four Gabor envelopes`;
4. parallel `Global MIONet coarse`, `Local U-Net residual`, and
   `Factorized spectral correction`;
5. summation, causality, free surface, and arbitrary saved frame.

Add a lower inset:

```text
Two observed complete fields at t0+2/f0 and t0+4/f0
        ↓
rank-4 LoRA in time/phase/source projections
        ↓
sealed adapter → blind future frames and z=20 m receiver line
```

Use solid blue borders for frozen parent blocks, green double borders for LoRA
targets, and magenta outlines for the two observations. Label tensor sizes
`201×201`, `12 channels`, and `r=4`. Do not show an elastic component.

- [ ] **Step 4: Draw the graphical abstract**

Use a 660:295 canvas ratio. Include only:

- velocity/source;
- two magenta observation snapshots;
- compact DCLP-NO+LoRA block;
- future sharp wavefront;
- near-surface trace overlay;
- one evidence-matched outcome phrase read from `claim_gate.json`.

If the dispersion gate fails, the phrase is:

```text
Physics-conditioned future wavefield prediction
```

If it passes:

```text
Reduced future phase and wavefront error
```

- [ ] **Step 5: Render and inspect**

Run:

```bash
/home/jiayh/miniconda3/bin/python scripts/make_tgrs_fig1.py \
  --output-base paper/tgrs_dclp_no/figures/fig1_architecture
/home/jiayh/miniconda3/bin/python scripts/make_tgrs_graphical_abstract.py \
  --claim-gate artifacts/tgrs_dclp_no/statistics/claim_gate.json \
  --output-base paper/tgrs_dclp_no/figures/graphical_abstract
```

Inspect PNGs at full resolution and PDFs at final column width. Require no
overlap, clipped label, rasterized text, or color-only semantic distinction.

### Task 5: Draw the numerical setup and dispersion figure

**Files:**
- Create: `scripts/make_tgrs_fig2.py`
- Create: `scripts/make_tgrs_fig3.py`

- [ ] **Step 1: Implement Fig. 2**

The left panel shows the 2 km by 2 km acoustic domain, pressure-free top,
three-sided CFS-CPML, source, 5 m physical grid, and 10 m stored grid. The
receiver line uses all 37 points at \(z=20\) m.

The right panel is a registered-time axis with:

- \(t_a=t_0+2/f_0\), observed;
- \(t_b=t_0+4/f_0\), observed;
- \(t_c=t_b+2/f_0\), blind;
- \(t_d=t_b+0.75(T_{\max}-t_b)\), blind;
- shaded blind interval \(t>t_b\).

Read coordinates from `protocol.yaml`; do not hard-code a second geometry.

- [ ] **Step 2: Implement Fig. 3**

Read only:

- `dispersion/analytic/phase_velocity.csv`;
- sealed per-record spatial-dispersion metrics;
- sealed receiver-phase metrics;
- paired confidence intervals.

Subpanels:

1. normalized phase velocity versus points per wavelength;
2. wavefront-radius error versus future time;
3. \(k\)-\(\omega\) ridge or dominant-wavenumber error;
4. receiver lag versus offset.

All learning-method curves use the same sample pairs. Add 95% confidence bands.

- [ ] **Step 3: Render and inspect**

```bash
/home/jiayh/miniconda3/bin/python scripts/make_tgrs_fig2.py \
  --protocol configs/tgrs_dclp_no/protocol.yaml \
  --output-base paper/tgrs_dclp_no/figures/fig2_setup
/home/jiayh/miniconda3/bin/python scripts/make_tgrs_fig3.py \
  --artifact-dir artifacts/tgrs_dclp_no \
  --output-base paper/tgrs_dclp_no/figures/fig3_dispersion
```

Expected: PDF and 600-dpi PNG for both figures.

### Task 6: Draw quantitative results, wavefields, receivers, and ablations

**Files:**
- Create: `scripts/make_tgrs_fig4.py`
- Create: `scripts/make_tgrs_fig5.py`
- Create: `scripts/make_tgrs_fig6.py`
- Create: `scripts/make_tgrs_fig7.py`

- [ ] **Step 1: Implement Fig. 4**

Use paired per-record data. Show:

- aggregate and family-wise future relative L2;
- error versus future time;
- spectral/phase error;
- bootstrap 95% confidence intervals.

Use points or compact violin/box summaries without bar-chart truncation.
Display all three families and the aggregate.

- [ ] **Step 2: Implement Fig. 5**

Read the pre-frozen `figure_cases.json`. For each family, render truth, strongest
baseline, frozen parent, and rank-4 LoRA at \(t_a,t_b,t_c,t_d\). Mark \(t_a,t_b\)
as observed and \(t_c,t_d\) as blind.

Rules:

```python
vmax = np.quantile(np.abs(truth), 0.995)
field_limits = (-vmax, vmax)
error_vmax = max(
    np.quantile(np.abs(method - truth), 0.995)
    for method in displayed_methods
)
```

Every method in a case shares the same field and error limits. Use a
zero-centered diverging colormap and meter coordinates.

- [ ] **Step 3: Implement Fig. 6**

Use exactly the fixed 37-receiver line. Display:

- three representative offsets selected by index before metrics are read;
- truth/baseline/parent/LoRA trace overlays;
- full truth gather;
- baseline residual gather;
- LoRA residual gather;
- offset-wise lag and coherence.

No receiver trace participates in LoRA fitting.

- [ ] **Step 4: Implement Fig. 7**

Combine:

- architecture ablation paired differences;
- LoRA rank 2/4/8;
- layer-target ablations;
- clean/20 dB/10 dB observations;
- error versus adaptation time/memory.

Mark the pre-registered primary method with a filled symbol and all ablations
with open symbols.

- [ ] **Step 5: Render and inspect**

```bash
for number in 4 5 6 7; do
  /home/jiayh/miniconda3/bin/python "scripts/make_tgrs_fig${number}.py" \
    --artifact-dir artifacts/tgrs_dclp_no \
    --output-base "paper/tgrs_dclp_no/figures/fig${number}"
done
```

Expected: every figure has PDF and 600-dpi PNG; wavefield panels additionally
export TIFF if the submission system requires it.

### Task 7: Generate tables directly from sealed statistics

**Files:**
- Create: `scripts/make_tgrs_tables.py`
- Generate: `paper/tgrs_dclp_no/tables/table1_protocol.tex`
- Generate: `paper/tgrs_dclp_no/tables/table2_main_results.tex`
- Generate: `paper/tgrs_dclp_no/tables/table3_ablation_efficiency.tex`

- [ ] **Step 1: Generate Table I**

Include dataset splits, domain/grid, time axis, boundary, source range,
observation times, receiver geometry, and parent checkpoint identity.

- [ ] **Step 2: Generate Table II**

Include eligible methods only, with:

- future field relative L2;
- wavefront-radius error;
- \(k\)-\(\omega\) or dominant-wavenumber error;
- receiver lag and coherence;
- phase correlation;
- paired 95% intervals.

Bold values by a deterministic minimum/maximum rule, not manual editing.

- [ ] **Step 3: Generate Table III**

Include structural ablations, LoRA rank/layers, trainable parameters, adaptation
seconds, inference seconds, and peak memory. Report rollback rate.

- [ ] **Step 4: Run table generation**

```bash
/home/jiayh/miniconda3/bin/python scripts/make_tgrs_tables.py \
  --artifact-dir artifacts/tgrs_dclp_no \
  --output-dir paper/tgrs_dclp_no/tables
```

Expected: three finite LaTeX tables with no `nan`, `inf`, or manual result
literals.

### Task 8: Create the IEEEtran manuscript scaffold

**Files:**
- Create: `paper/tgrs_dclp_no/main.tex`
- Create: eight section files.

- [ ] **Step 1: Create `main.tex`**

```latex
\documentclass[journal]{IEEEtran}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{siunitx}
\usepackage[caption=false,font=footnotesize]{subfig}
\usepackage{xcolor}
\usepackage{url}
\graphicspath{{figures/}}

\begin{document}
\title{\input{generated/title.tex}}
\author{Yang~Cui}
\maketitle

\begin{abstract}
\input{sections/abstract}
\end{abstract}
\begin{IEEEkeywords}
Acoustic wave equation, neural operator, numerical dispersion,
parameter-efficient adaptation, seismic modeling.
\end{IEEEkeywords}

\input{sections/introduction}
\input{sections/problem_reference}
\input{sections/method}
\input{sections/lora}
\input{sections/protocol}
\input{sections/results}
\input{sections/discussion}
\input{sections/conclusion}

\bibliographystyle{IEEEtran}
\bibliography{references}
\end{document}
```

The scaffold uses the project’s recorded author, Yang Cui. Confirm the complete
author order, affiliations, ORCIDs, corresponding author, funding, and
acknowledgments before the final submission build. If current TGRS review is
single blind, do not anonymize the confirmed list.

- [ ] **Step 2: Create the section files**

Each file starts with its exact heading and evidence purpose:

```latex
% sections/problem_reference.tex
\section{Problem Formulation and Numerical Reference}
\label{sec:problem}
```

Do the same for the approved eight-section structure. The abstract contains no
number until final tables are sealed.

- [ ] **Step 3: Compile the empty structural draft**

```bash
cd paper/tgrs_dclp_no
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Expected: PDF is created and the log has no undefined control sequence.

### Task 9: Write the manuscript from evidence

**Files:**
- Modify all section files.
- Create `paper/tgrs_dclp_no/generated/title.tex`.
- Create `paper/tgrs_dclp_no/claims_audit.md`.

- [ ] **Step 1: Generate the title from the claim gate**

If `dispersion_suppression_supported=true`:

```text
Dispersion-Controlled Local-Propagation Neural Operator With Two-Snapshot LoRA Adaptation for Acoustic Wavefield Modeling
```

Otherwise:

```text
Physics-Conditioned Local-Propagation Neural Operator With Two-Snapshot LoRA Adaptation for Acoustic Wavefield Modeling
```

- [ ] **Step 2: Draft the methods before results**

Write complete paragraphs for:

- 2-D acoustic equation, source, free surface, CPML;
- LWC-84 reference and 401-to-201 restriction;
- medium/source encoders and 12-channel propagation bundle;
- global coarse plus local residual plus spectral corrector;
- two-cycle/four-cycle LoRA protocol and loss;
- sealed data boundary and 37 receivers.

Use equations for the field decomposition, retarded time, LoRA update, and
two-snapshot objective. Do not claim a saved-frame PDE loss.

- [ ] **Step 3: Draft results from generated tables**

Every numerical sentence cites a generated table/figure and can be traced to a
JSON/CSV row. Include:

- full test-ID performance;
- family and late-time behavior;
- direct dispersion metrics;
- receiver results;
- LoRA future benefit or failure and rollback rate;
- ablations and efficiency;
- negative results, including high-frequency limitations.

- [ ] **Step 4: Draft introduction and discussion**

The introduction states four contributions from the approved design and cites
primary literature. The discussion separates:

- numerical-reference dispersion;
- learned approximation/phase error;
- limitations from travel-time quality;
- acoustic-only scope;
- absence of receiver supervision.

- [ ] **Step 5: Write abstract and conclusion last**

The abstract contains problem, method, exact two-snapshot protocol, test scale,
two or three central sealed values, and evidence-matched conclusion. It does
not say “eliminates dispersion” or “state of the art.”

- [ ] **Step 6: Run the claims audit**

`claims_audit.md` has one row per quantitative or comparative claim:

```text
claim | section | artifact | metric key | confidence interval | permitted by gate
```

Any unsupported row blocks compilation in `build_tgrs_paper.py`.

### Task 10: Write supplement, reproducibility README, and cover letter

**Files:**
- Create: `paper/tgrs_dclp_no/supplement/supplement.tex`
- Create: `paper/tgrs_dclp_no/README.md`
- Create: `paper/tgrs_dclp_no/cover_letter/cover_letter.tex`

- [ ] **Step 1: Build the supplement**

Include:

- complete architecture and LoRA target list;
- all configuration identities;
- family/OOD/noise/ablation tables;
- leakage and future-mutation tests;
- extra wavefields and all 37 traces;
- numerical-dispersion derivation;
- compute and carbon/hardware disclosure if requested by current guidance.

- [ ] **Step 2: Write the exact reproducibility commands**

README sections:

1. environment and Python path;
2. protected parent identity;
3. protocol audit;
4. LoRA adaptation;
5. sealed future evaluation;
6. baseline commands;
7. statistics and claim gate;
8. figure/table generation;
9. LaTeX build;
10. artifact hashes.

- [ ] **Step 3: Write the cover letter**

The cover letter contains:

- manuscript title selected by the claim gate;
- TGRS relevance;
- four evidence-matched contributions;
- acoustic-only scope;
- originality and concurrent-submission statement for author confirmation;
- graphical abstract and supplementary-material statement.

It contains no unverified superlative.

### Task 11: Implement the final submission verifier

**Files:**
- Create: `tests/tgrs_dclp_no/test_submission_verifier.py`
- Create: `scripts/verify_tgrs_submission.py`

- [ ] **Step 1: Write failing package-audit tests**

Build a temporary minimal paper tree. Assert the verifier fails for:

- missing Fig. 1–7;
- missing claim gate;
- `[UNRESOLVED]` or `[CITATION NEEDED]`;
- “dispersion-free”;
- fewer than 37 receivers;
- observation count other than 2;
- missing PDF;
- unresolved LaTeX citation warning.

Assert a complete synthetic package passes.

- [ ] **Step 2: Implement checks**

The verifier:

```python
required_figures = tuple(f"fig{index}" for index in range(1, 8))
forbidden_text = (
    "[UNRESOLVED]",
    "[CITATION NEEDED]",
    "dispersion-free",
    "eliminates numerical dispersion",
)
```

It reads protocol, claim gate, LaTeX sources, build log, figure files, generated
tables, and artifact identities. It calls:

```bash
pdfinfo paper/tgrs_dclp_no/main.pdf
pdffonts paper/tgrs_dclp_no/main.pdf
```

It requires embedded/subset fonts and zero missing figures/citations.

- [ ] **Step 3: Run verifier tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_submission_verifier.py
```

Expected: all tests pass.

### Task 12: Compile, visually inspect, and seal the submission

**Files:**
- Generate: `paper/tgrs_dclp_no/main.pdf`
- Generate: `paper/tgrs_dclp_no/supplement/supplement.pdf`
- Generate: `paper/tgrs_dclp_no/cover_letter/cover_letter.pdf`
- Generate: `paper/tgrs_dclp_no/submission_manifest.json`

- [ ] **Step 1: Build all documents**

```bash
/home/jiayh/miniconda3/bin/python scripts/build_tgrs_paper.py \
  --paper-dir paper/tgrs_dclp_no \
  --artifact-dir artifacts/tgrs_dclp_no
```

Expected: all PDFs compile without undefined reference or citation warnings.

- [ ] **Step 2: Render every page for visual review**

```bash
mkdir -p artifacts/tgrs_dclp_no/paper_review/main
pdftoppm -png -r 160 \
  paper/tgrs_dclp_no/main.pdf \
  artifacts/tgrs_dclp_no/paper_review/main/page
```

Inspect every rendered page for clipped figures, unreadable labels, bad floats,
blank space, table overflow, and inconsistent symbols. Record pass/fail for
each page in `paper_review.json`.

- [ ] **Step 3: Run the final verifier**

```bash
/home/jiayh/miniconda3/bin/python scripts/verify_tgrs_submission.py \
  --paper-dir paper/tgrs_dclp_no \
  --artifact-dir artifacts/tgrs_dclp_no \
  --require-figures 7 \
  --require-receiver-count 37 \
  --require-observed-snapshots 2
```

Expected: exit code 0 and `submission_manifest.json` has `status=verified`.

- [ ] **Step 4: Stop the brainstorming visual companion**

After the package is verified:

```bash
bash /home/jiayh/.codex/skills/brainstorming/scripts/stop-server.sh \
  /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/project/.superpowers/brainstorm/2167315-1784904870
```

Expected: the server PID exits cleanly.
