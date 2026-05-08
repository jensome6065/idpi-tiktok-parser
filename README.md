# TikTok AIGC Detection Project - New Student Handoff Guide

This repository contains the full research workflow for measuring AI-generated content (AIGC) signals on TikTok using:

- platform metadata labels,
- creator self-disclosure (hashtags + captions),
- and manual labels (for the curated dataset).

If you are joining the project now, this README is designed to let you run the pipeline immediately and modify it safely.

## 1) Project at a Glance

### Research question
How often does TikTok label videos as AI-generated, how often do creators disclose AI usage, and how well these signals agree.

### Main datasets
- `metadata_combined.jsondir.csv`: large-scale metadata sample (random/API-style dataset).
- `tiktok_database.csv`: manual sheet with `LINK` + human labels.
- `tiktok_database_metadata/`: raw JSON pulled from TikTok pages for links in `tiktok_database.csv`.

### Core outputs
- `research_outputs_jsondir/`: prevalence/agreement tables + summary figures + `findings_summary.md`.
- `paper_figures/`: publication and poster figure pack.
- `RESEARCH_WRITEUP.md`: narrative report with methods/findings.

## 2) Repository Map (What each script does)

- `fetch_tiktok_database_json.py`  
  Pulls TikTok embedded JSON using Selenium and saves one JSON per video + a fetch report CSV.

- `tiktok_database_parser.py`  
  Merges manual labels (`tiktok_database.csv`) with fetched JSON metadata and creates:
  - `tiktok_database_parsed.csv`
  - `tiktok_database_combined.csv`

- `enrich_signals.py`  
  Adds derived signal columns (hashtags, caption regex, platform signal, agreement metrics, era split, normalized engagement).

- `research_findings_report.py`  
  Builds reproducible analysis tables/figures and writes findings markdown for any enriched CSV.

- `generate_paper_figures.py`  
  Creates the paper/poster figure pack under `paper_figures/` and `paper_figures/poster_ready/`.

- `RESEARCH_WRITEUP.md`  
  Main narrative writeup (methods, results, discussion, limitations).

- `claude_slide_assets/README_ASSET_CHECKLIST.txt`  
  Slide-asset checklist and presentation tracking notes.

## 3) Quick Start (Run this first)

From repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install pandas matplotlib selenium
```

You will also need:
- Google Chrome installed
- a matching ChromeDriver on PATH (or Selenium Manager-supported setup)

Check toolchain:

```bash
python3 --version
python3 -c "import pandas, matplotlib, selenium; print('ok')"
```

## 4) End-to-End Pipeline

## A) Fetch metadata for manual-link dataset

```bash
python3 fetch_tiktok_database_json.py \
  --sheet tiktok_database.csv \
  --outdir tiktok_database_metadata \
  --out-report tiktok_database_metadata_fetch_report.csv \
  --headless 1 \
  --sleep 7.0
```

Notes:
- If TikTok blocks/loads slowly, increase `--sleep`.
- Use `--limit` for test runs.

## B) Parse + combine with manual labels

```bash
python3 tiktok_database_parser.py \
  --sheet tiktok_database.csv \
  --metadata-dir tiktok_database_metadata \
  --fetch-report tiktok_database_metadata_fetch_report.csv \
  --out-parsed tiktok_database_parsed.csv \
  --out-combined tiktok_database_combined.csv \
  --cutoff 2024-02-01
```

## C) Enrich signals (for any combined CSV)

Example for the main metadata file:

```bash
python3 enrich_signals.py \
  --input metadata_combined.jsondir.csv \
  --output metadata_enriched.jsondir.csv \
  --cutoff 2024-02-01 \
  --report
```

## D) Generate analysis tables + findings summary

```bash
python3 research_findings_report.py \
  --input metadata_enriched.jsondir.csv \
  --outdir research_outputs_jsondir
```

This writes CSV tables, PNG figures, and `research_outputs_jsondir/findings_summary.md`.

## E) Build paper/poster figures

```bash
python3 generate_paper_figures.py
```

Outputs:
- `paper_figures/*.png`
- `paper_figures/poster_ready/*.png`
- `paper_figures/figure_manifest.csv`

## 5) How to Modify the Project

### Update AI hashtag lexicon / caption rules
Edit `enrich_signals.py`:
- `AI_HASHTAGS` set
- `CAPTION_AI_REGEX`

Then rerun enrichment + reporting steps.

### Change pre/post AI cutoff date
Use `--cutoff YYYY-MM-DD` in:
- `enrich_signals.py`
- `tiktok_database_parser.py`

### Add a new detection signal
Recommended pattern:
1. Add column in `enrich_signals.py` (e.g., `signal_visual_model` or other signal).
2. Include agreement logic in enrichment.
3. Add prevalence/agreement rows in `research_findings_report.py`.
4. Add figure or table in `generate_paper_figures.py` if needed.

### Update figure styling
Edit plotting functions in `generate_paper_figures.py`.  
Poster-specific formatting lives in:
- `_apply_poster_style()`
- `make_monthly_label_rate_figure_poster()`
- `make_summary_stats_platform_poster()`

## 6) Data + Output Conventions

- Large generated outputs are kept as CSV/PNG artifacts in:
  - `research_outputs_jsondir/`
  - `paper_figures/`
  - `claude_slide_assets/`
- Raw HTML/JSON scrape artifacts are in:
  - `html_dumps/`
  - `tiktok_database_metadata/` (if generated)

When rerunning analysis, prefer writing to a versioned output directory (for reproducibility), e.g.:
- `research_outputs_YYYYMMDD/`

## 7) Reproducibility Checklist for New Team Members

Before claiming results:
- Confirm Python package versions in your environment.
- Re-run enrichment and report scripts from raw/combined inputs.
- Compare core metrics:
  - platform prevalence,
  - creator-tag prevalence,
  - disclosure gap,
  - positive agreement.
- Verify figures regenerated in `paper_figures/` without runtime warnings.

## 8) Background Reading + Tutorials

### Python data stack
- [Pandas Getting Started](https://pandas.pydata.org/docs/getting_started/index.html)
- [Matplotlib Pyplot Tutorial](https://matplotlib.org/stable/tutorials/pyplot.html)
- [Python `re` (Regex) docs](https://docs.python.org/3/library/re.html)

### Web scraping/runtime tools
- [Selenium with Python docs](https://selenium-python.readthedocs.io/)
- [Selenium official documentation](https://www.selenium.dev/documentation/)

### Methods/statistics
- [Wilson score interval overview](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Wilson_score_interval)
- [Pandas groupby user guide](https://pandas.pydata.org/docs/user_guide/groupby.html)

### Domain context
- [TikTok newsroom: AI-generated content labeling policy updates](https://newsroom.tiktok.com/)
- [OECD AI policy observatory](https://oecd.ai/)

## 9) How This Connects to the Report

Primary report:
- `RESEARCH_WRITEUP.md`

Recomputed findings:
- `research_outputs_jsondir/findings_summary.md`

Figure index for paper/poster:
- `paper_figures/figure_manifest.csv`

If you update methods, update these together in one pass:
1. script changes,
2. regenerated outputs,
3. report text in `RESEARCH_WRITEUP.md`,
4. slide assets/checklist in `claude_slide_assets/README_ASSET_CHECKLIST.txt` (if relevant).

## 10) Common Issues

- **Selenium fails to launch Chrome**
  - Confirm Chrome install and ChromeDriver compatibility.
  - Retry with updated Selenium and local browser update.

- **Many `no_embedded_json` rows**
  - Increase `--sleep` and rerun failed URLs.
  - Inspect blocked pages manually.

- **Missing columns in analysis input**
  - `research_findings_report.py` is robust to multiple schemas, but best results come from enriched CSVs produced by `enrich_signals.py`.

- **Very low AI prevalence**
  - This is expected in current samples; use confidence intervals + positive agreement in interpretation.

## 11) Recommended Next Extensions

- Add a visual-model signal column from external classifiers and evaluate agreement with platform/manual labels.
- Add formal statistical tests for engagement comparisons (distribution-aware tests with effect sizes).
- Add language-aware hashtag/caption lexicons for non-English disclosures.
- Add a small `requirements.txt` or `pyproject.toml` lockfile for stricter reproducibility.

