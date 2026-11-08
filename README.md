# Promo Finder v2 — reproducible script-only microdrama promo discovery

**What it does:** turns a ZIP (or TXT) of timestamped screenplays into ranked 20–60-second *suggested* promo cuts with episode number, exact segment start/end, first-three-second hook, end beat, explanation, warnings and downloadable raw outputs. There are no edited videos and scores are not predictions of conversion.

**What changed from v1:** general scoring (no characters/show names in code), first-three-second hook evaluation, explicit warnings for incomplete openings / rhetorical questions / resolved endings / exposition / cross-scene jumps, stricter input checks, TF-IDF near-duplicate filtering, protection against stale outputs when settings or inputs change, and an optional editor-review CSV. Continuous one-episode cuts only. The model is not trained or fine-tuned; it is deterministic NLP and heuristics. A separately included optional OpenAI-compatible adapter is not used for the offline results.

## What's inside

- `prototype/app.py` — Streamlit interface
- `prototype/promo_finder.py` — offline scoring/ranking and CLI analysis engine
- `prototype/optional_llm.py` — optional API adapter, **not used** in offline results
- `prototype/requirements.txt` — dependencies
- `demo_input/demo_episode_001.txt` — fictional museum demonstration
- `demo_input/unseen_space_show/` — **completely different fictional show**, two episodes
- `testing/test_promo_finder.py` — unit and regression tests
- `WINDOWS_START_HERE.md` — detailed Windows + VS Code commands
- `.gitignore` — prevents accidental committing of typical local output and private inputs

## Local setup — Windows PowerShell inside VS Code

Open the extracted **project root** in VS Code. It should contain `prototype`, `demo_input`, `testing`, `README.md` and `WINDOWS_START_HERE.md`. Then open **Terminal → New Terminal**.

```powershell
py --version
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\prototype\requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s .\testing -v
.\.venv\Scripts\python.exe -m streamlit run .\prototype\app.py
```

Open `http://localhost:8501`. The local address **cannot be submitted as the prototype link**. After final checks, deploy to Streamlit Community Cloud or submit a public GitHub repository with clear run instructions.

## Steps in the app

1. Upload the **original** timestamped screenplay ZIP into the local app. Do not extract the original ZIP first.
2. Set `Number of ranked promos = 20` and `Maximum promos from one episode = 1`.
3. Press **Analyze all uploaded episodes**. `Episodes parsed` counts files; `Ranked suggestions` counts clips selected, so 50 and 20 are different correct values.
4. Examine top promo starts/ends, score breakdown, warning list, and contextual dialogue.
5. Download all six raw files. If you change the ZIP or sliders, press **Analyze** again.
6. Enter **human-editor** verdicts if desired; download that separate CSV (never modify raw model outputs).

## Command-line example — with no UI

```powershell
.\.venv\Scripts\python.exe .\prototype\promo_finder.py "C:\path\to\company_scripts.zip" --out .\my_outputs --count 20 --per-episode 1
```

The output folder contains:

| File | Purpose |
|---|---|
| `promos_ranked.json` | Full ranked output with reasons, warnings, scoring fields and exact segment times |
| `promos_ranked.csv` | Easy-to-read list of selected cuts |
| `episode_coverage.csv` | File-by-file parser coverage and warnings |
| `candidate_scores_top3_per_episode.csv` | Top three candidate scores for every episode |
| `editor_review_template.csv` | Blank human assessment rubric (not fabricated reviews) |
| `run_summary.json` | Count checks and limitations |

## Data safety and deployment

Kuku's original 50-episode pack is **confidential**. Do **not** commit its `.zip`, source `.txt` files, generated dialogue excerpts, or raw company outputs to a public GitHub repository. Public demo code may include only fictional `demo_input` samples. Share company-based raw results privately with the authorized hiring reviewers, according to their submission instructions. Uploaded scripts are processed in memory/on temporary local storage, and are not bundled with code. When deploying an open app, do not assume third-party hosting is private; use fictional demo inputs for public tests.

**Never** commit API keys or secrets. No API key is required for this prototype. If you use an external model in the future, confirm you have authorization to transmit script excerpts to it.

## Evaluation and limits

- Checks **technical validity** (durations 20–60s, supported timestamps, script parsing, clip boundaries) automatically.
- Ranking **quality remains subjective**. Read the scene and check whether a new viewer can understand it. Video, audio and social-media performance cannot be verified from scripts.
- Known failure modes: urgent-looking but mundane words; rhetorical questions misclassified as cliffhangers; dialogue describing events rather than showing them; visual-only scene hooks; changing script format; and stitched cross-episode promos not supported.
- Scores have no calibrated meaning and **cannot** be compared numerically with v1 scores (different scales and scoring functions).
- No paid models were called to generate the offline raw outputs.
- Changing the parser for a new format is expected; compatibility with Kuku-style timed text does not imply support for PDFs, MP4s or untimed stories.

