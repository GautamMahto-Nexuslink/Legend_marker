# legend_marker

Replace the generic class names a detector gives map icons with the **real names
printed in that map's own legend**.

A detector called an icon `other_icon 0.92`. The legend next to it says
`Barn`. This pipeline reads the legend, matches every map icon to a legend
entry, and renames it — falling back on a curated database of known icons when
the legend cannot decide.

```
map.png + legend.png  ──►  results.json          renamed icons + boxes
                           map_annotated.png     colour-coded per decision
                           map_crops/*.txt       why each icon got its name
```

---

## Install

```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr        # only for --ocr-engine tesseract
```

## Run

```bash
# one map
python3 legend_marker.py \
    --map    /path/to/map.jpg \
    --legend /path/to/legend.jpg \
    --api-key XXX --project plotmymap-icon-lqf56 --version 1 \
    --output-dir output/my_run -v

# a whole folder (edit the CONFIG block at the top first)
python3 batch_run.py
```

New here? Read **[WORKFLOW.md](WORKFLOW.md)** — it gives the order to run things
in and the things that bite.

---

## How a name is decided

Every map icon goes through four stages. The colour is what you see on
`map_annotated.png`, and every stage's answer is written to the crop's `.txt`.

| colour | stage | what it means |
|---|---|---|
| 🟠 orange | detector class | nothing below was confident — the original class is kept |
| 🔵 blue | **known-icon DB** | identified against `icons_glyph_db.npz` (~4900 curated icons) |
| 🟢 green | **legend / OCR** | matched to a legend entry by glyph similarity |
| 🟣 magenta | **DB + legend agree** | the DB identified it *and* this map's legend has a same/nearby label, so the legend's own wording is used (`Observation Area` → the legend's `Overlook`) |

The blue stage does **not** use a plain hash cutoff: it prefilters on three
hashes, re-scores the shortlist with template correlation + ORB on a stored
64×64 glyph, lets a class's exemplars vote per semantic group, and accepts only
a clear winner. Measured leave-one-out on the current 7537-icon database:
**85% of icons identified at 96% accuracy** — a fixed `hamming<=10` cutoff
manages 2%. Re-measure any time with `python3 benchmark_icon_db.py`.

---

## Scripts

### Run the pipeline

| script | what it does |
|---|---|
| **`legend_marker.py`** | CLI entry point for **one** map + legend. Thin shim over `legend_pipeline/`, so `import legend_marker as lm` also gives you the whole API. |
| **`batch_run.py`** | Runs the pipeline over a folder of maps, pairing each with the legend of the same filename. Edit the `CONFIG` dict at the top — no command-line flags. One failure never stops the batch; writes `batch_summary.json`. |
| **`save_icons.py`** | Same inputs as `batch_run.py`, but keeps **only the final icons**, collated as `OUTPUT_FOLDER/<ClassName>/<mapstem>_1.png`. This is how the icon dataset gets grown. |
| **`verify_legend.py`** | Replays a saved `legend_roboflow_raw.json` through the current code — no network, no API key. Use it to check a code change without paying for inference. |

### Build the databases the pipeline reads

| script | builds | when to re-run |
|---|---|---|
| **`build_icon_db.py`** | `icons_glyph_db.npz` — each icon's 64×64 glyph + pHash/dHash/aHash. Powers the blue stage. | whenever icons are added to the crops folder |
| **`build_class_synonyms.py`** | `icon_class_synonyms.json`, `icon_class_related.json`, `icon_class_neighbors.json`, `class_to_canonical.json`, `canonical_to_classes.json`, `search_index.json` | after editing `icon_class_synonyms.py` |
| **`save_phash.py`** | `icons_phash.json` + `icons_phash_flat.json` — the older hash-only DB. Still read as a fallback when the `.npz` is missing. | rarely; superseded by `build_icon_db.py` |

### Curate the icon dataset

| script | what it does |
|---|---|
| **`sort_icons_by_phash.py`** | Sorts loose icon images into per-class folders using the *same* template+ORB matcher the pipeline uses (not raw pHash). Has `--copy`, `--unmatched-dir`, `--shortlist`. |
| **`split_folders_alpha.py`** | Splits an icon tree into `A-H` / `I-P` / `Q-Z` groups by filename, keeping the sub-folder structure — for sharing review work out. |
| **`change_name_imagee.py`** | Turns a Roboflow export name (`Alamo lake_jpg.rf.HkJN0….jpg`) back into its clean original (`Alamo_lake.jpg`). |
| **`icons_to_pdf.py`** | Builds a PDF contact sheet of icon crops grouped by source image then class — for eyeballing a whole dataset quickly. |

### Measure and test

| script | what it does |
|---|---|
| **`benchmark_icon_db.py`** | Leave-one-out benchmark of the blue stage over the whole icon DB. Prints coverage / accuracy / yield per strategy, so a threshold change can be judged on numbers. `--degrade` distorts queries to imitate real map rendering. |
| **`test_icon_db.py`** | Probes the blue stage with real crops from ~174 classes, plus its refusal gates and the legacy-JSON fallback. |
| **`test_synonym_decision.py`** | The synonym/related graph and every branch of the final naming decision. |
| **`test_label_truncation.py`** | That a legend label is never read truncated (`arking Area`) when a detection box overlaps it. |

Run all three tests with `for t in test_*.py; do python3 $t; done`.

### The package

`legend_pipeline/` holds the implementation, one stage per module:

| module | role |
|---|---|
| `config.py` | **every** tunable knob, with the reasoning inline. Start here. |
| `cli.py` | argparse → `PipelineConfig` → `main()` |
| `pipeline.py` | orchestration: the 6 steps and the final naming decision |
| `detector.py` | Roboflow inference (steps 1 & 5) |
| `ocr.py` | Tesseract / EasyOCR / PaddleOCR → cleaned text boxes (step 2) |
| `matching.py` | icon ↔ label matching, false-positive and glyph filters (step 3) |
| `signatures.py` | glyph segmentation, ORB, pHash, and the weighted matcher (steps 4 & 6) |
| `icon_db.py` | the known-icon database and its retrieval strategy (blue stage) |
| `synonyms.py` | canonical-name lookup + the nearby-concept graph |
| `orientation.py` | corrects legends/maps scanned sideways |
| `visualization.py` | the annotated debug images, and the colour per decision |
| `reporting.py` | the per-crop `.txt`: what every stage called the icon |
| `containers.py`, `deps.py`, `utils.py`, `timing.py` | dataclasses, optional imports, helpers, per-step timings |

---

## Data files

| file | contents |
|---|---|
| `icons_glyph_db.npz` | every icon × (glyph + 3 hashes + class) — the blue stage's database (currently 7537 icons / 364 classes) |
| `icon_class_synonyms.py` | **the source of truth you edit**: the canonical groups + the related-concept graph |
| `icon_class_synonyms.json` | canonical name → synonyms / alternate spellings / search terms |
| `icon_class_related.json` | canonical name → neighbouring canonical names |
| `icon_class_neighbors.json` | one entry per original class (all 292): its group, same-meaning terms, nearby classes |
| `class_to_canonical.json`, `canonical_to_classes.json`, `search_index.json` | lookup tables generated alongside |
| `icons_phash_flat.json`, `icons_phash.json` | legacy hash-only DB (fallback) |
| `improvent_points.txt` | running list of known issues and their fixes |

## Output of a run

```
output/<run>/
    legend_detections_raw.png   what the detector saw on the legend
    legend_ocr_text.png         what the OCR read, and where
    legend_annotated.png        icon → label assignment (check this first)
    legend_crops/               one crop + .txt per legend icon
    legend_results.json         legend icon → name
    map_detections_raw.png      raw detector output on the map
    map_annotated.png           final names, colour-coded per stage
    map_crops/                  one crop + .txt per map icon
    map_results.json            the deliverable: class, bbox, score, method
    *_roboflow_raw.json         raw inference responses (replayable)
```
