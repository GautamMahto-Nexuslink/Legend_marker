# Workflow

How to actually run this, in order, and the things that will bite you.
For what each script is, see [README.md](README.md).

---

## 0. Once per machine

```bash
pip install -r requirements.txt
sudo apt-get install tesseract-ocr          # only if you use --ocr-engine tesseract
python3 -c "import legend_marker; print('ok')"
```

You need a **Roboflow API key** and project. Either put them on the command
line, or export them so they stay out of your shell history:

```bash
export ROBOFLOW_API_KEY=...
export ROBOFLOW_PROJECT=plotmymap-icon-lqf56
```

---

## 1. Check the databases exist

The pipeline reads two generated artefacts. Both are committed, so normally you
skip this — but rebuild after **any** change to the icon dataset or the synonym
source.

```bash
# blue stage: 4894 icons -> icons_glyph_db.npz   (~80 s)
python3 build_icon_db.py /home/nls34/Downloads/Dataset_lengend_marker_viewer/IMP/Merged_Save_Crops -v

# semantic groups: icon_class_synonyms.py -> the JSONs   (instant)
python3 build_class_synonyms.py
```

`build_class_synonyms.py` prints `all classes mapped ✓`. If it lists
**UNMAPPED CLASSES** it exits non-zero: a class folder exists that no group
covers. Add it to `icon_class_synonyms.py` and re-run.

> **Outstanding right now:** the crops folder has grown to 364 classes and **60
> of them have no semantic group yet** (`Accessible restrooms`, `Boat Launch
> Hand`, `Canoe Kayak Launch`, `Dumping Station`, …). They still work — an
> unmapped class just becomes its own group — but they cannot merge with the
> class they duplicate, so `Accessible restrooms` competes with `Restrooms`
> instead of voting with it. Run `python3 build_class_synonyms.py` to list all
> 60, then extend `ICON_CLASS_SYNONYMS` / `RELATED_CONCEPTS` in
> `icon_class_synonyms.py`.

---

## 2. Run one map first

Never start a batch before a single map looks right.

```bash
python3 legend_marker.py \
    --map    /path/to/map.jpg \
    --legend /path/to/legend.jpg \
    --api-key "$ROBOFLOW_API_KEY" --project "$ROBOFLOW_PROJECT" --version 1 \
    --ocr-engine easyocr \
    --output-dir output/sanity_check -v
```

Then look at the output **in this order** — each answers a different question:

| # | file | question it answers |
|---|---|---|
| 1 | `legend_ocr_text.png` | did OCR read the legend labels at all? |
| 2 | `legend_annotated.png` | is each legend icon paired with the *right, whole* label? |
| 3 | `map_annotated.png` | what did each map icon end up called, and via which stage (colour)? |
| 4 | `map_crops/NNN_*.txt` | for any icon that looks wrong: what every stage said, and why the winner won |

Step 2 is the one that matters most. **If the legend mapping is wrong, nothing
downstream can be right** — the legend is the naming authority for the map.

---

## 3. Then run the batch

```bash
# edit the CONFIG dict at the top of batch_run.py: INPUT_FOLDER, LEGEND_FOLDER,
# OUTPUT_FOLDER, API_KEY, PROJECT
python3 batch_run.py
```

* Maps are paired with legends **by filename** (exact stem first, then a
  normalised fallback that ignores spaces/dots/underscores). A map with no
  matching legend is skipped and reported.
* `SKIP_EXISTING: True` means a re-run only processes maps that failed or are
  new — so you can stop and resume.
* Set `LIMIT: 5` to smoke-test the config on five maps before committing to
  hundreds.
* Read `batch_summary.json` at the end for per-map success/failure.

---

## 4. Grow the icon dataset (optional loop)

The blue stage gets better the more curated icons it has. The loop:

```bash
python3 save_icons.py                          # harvest final icons into <Class>/ folders

# file the loose ones against the database — dry run first, always
python3 sort_icons_by_db.py <folder of loose icons> --out sorted_icons --dry-run
python3 sort_icons_by_db.py <folder of loose icons> --out sorted_icons \
        --copy --unmatched-dir needs_review

python3 icons_to_pdf.py sorted_icons --out icons_report.pdf   # eyeball the result
# ... fix any mis-sorted icons by hand, and label needs_review/ ...
python3 build_icon_db.py <crops folder> -v                    # rebuild the DB
python3 benchmark_icon_db.py --limit 800                      # confirm it improved
```

`sort_icons_by_db.py` uses the same identification as the pipeline, so an icon
lands under the name the pipeline would give it. `--margin` is the dial: raise it
to sort fewer icons but misfile fewer of them, and send the rest to
`--unmatched-dir` for a human. Measured on 26 icons distorted to imitate fresh
map crops: **85% sorted, all of them correctly**, 4 left for review.

`benchmark_icon_db.py` is the arbiter for any threshold change. Current
baseline, leave-one-out on the 7537-icon / 364-class database:

| strategy | coverage | accuracy |
|---|---|---|
| `hamming<=10` (the old stage) | 2.2% | 64% |
| **glyph re-score + group vote (now)** | **84.6%** | **95.5%** |

Two different metrics, both honest — don't be surprised when they disagree:

* `benchmark_icon_db.py` samples **icons**, so common classes carry more weight.
  This is what a real map sees: ~85% / ~95%.
* `test_icon_db.py` probes **one exemplar per class**, so a class with 3 crops
  counts as much as one with 200. Deliberately harder: ~77% / ~88%.

---

## Points to remember

**Rebuild the databases after changing their sources.** New icon folders need
`build_icon_db.py`; edits to `icon_class_synonyms.py` need
`build_class_synonyms.py`. Nothing warns you — the pipeline just keeps using the
old file.

**`icon_class_synonyms.py` is the file you edit.** The `.json` files are
generated. Editing the JSON directly gets overwritten on the next build.

**Config lives in three places, and they don't share defaults.**
`legend_pipeline/config.py` holds every knob with the reasoning inline — that is
the reference. `legend_marker.py` exposes a subset as CLI flags. `batch_run.py`
and `save_icons.py` each have their own `CONFIG` dict; anything they don't
forward silently uses the `config.py` default. (`save_icons.py` in particular
does not forward the icon-DB or synonym knobs — it inherits the defaults, which
is usually what you want, but you cannot tune them from its CONFIG.)

**Absolute paths in the config dicts are machine-specific.** `config.py`,
`batch_run.py` and `save_icons.py` all contain `/home/nls34/...` defaults. On
another machine, override them.

**The legend must be a crop of the legend, not the whole map.** `--legend` is a
tight crop; `--map` is the full page.

**Sideways pages are handled, but cost time.** `auto_rotate` probes four
orientations with OCR on the legend, then reuses that angle for the map. Pass
`--no-auto-rotate` when you know the inputs are upright.

**One dial trades coverage against mistakes on the blue stage:**
`--icon-db-margin` / `ICON_DB_MARGIN`. `0.04` ≈ 89% coverage at 96% accuracy,
`0.06` (default) ≈ 86% at 97%, `0.08` ≈ 85% at 98%. Icons the DB refuses are not
lost — they fall through to the green legend stage.

**A *related* rename needs visual proof.** The DB saying `Trash` will only become
the legend's `Campground` if that legend entry is also the best visual match and
clears the score gates. That gate exists because it once produced
`Trash → Campground` on weak evidence; `--weak-related-ok` removes it (unsafe).

**`--ocr-engine easyocr` is what the batch configs use.** Tesseract is the code
default and is faster, but EasyOCR reads small legend text more reliably. Pick
one and keep it fixed while comparing runs.

**Roboflow inference costs money and time.** To test a code change against a
legend you already ran, replay it offline instead:

```bash
python3 verify_legend.py <legend_image> output/<run>/legend_roboflow_raw.json
```

**Run the tests after touching matching, naming or the icon DB.** They need no
API key and take seconds:

```bash
for t in test_*.py; do echo "== $t"; python3 "$t" | tail -1; done
```

**Keep notes in `improvent_points.txt`.** Known issues and their intended fixes
are tracked there; it is the first place to look when something reads oddly.
