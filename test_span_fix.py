"""
Test the span fix logic without running any models.
Compares target_director_views computed two ways for each structure:
  1. Pre-stored views from dataset (all size=1, original paper behavior)
  2. Live recomputed with spans (size=2 for dominoes, rebuttals behavior)

  How stored_large is computed

The dataset file structures_dataset_20.json has a field called director_views stored directly in it — it was pre-generated when the dataset was built and saved as-is. The script reads that field and counts how many cells have size: 2 in it.

Each director view looks like this in the JSON:

"D1": {
  "row_0": [{"color": "yellow", "size": 1}, {"color": "blue", "size": 1}, ...],
  "row_1": [...],
  "row_2": [...]
}
count_large just scans every cell across D1/D2/D3, all rows, and counts where size == 2.

What zero means

Every single cell in the stored dataset has size: 1. Not a single domino was ever encoded as size=2 when the dataset was generated. So when the original paper ran, directors received their target view with all blocks described as size 1 — even structures where two adjacent cells are physically spanned by one large block.

The root cause: get_director_views_fn was called without passing spans when the dataset was built, so it defaulted to size=1 for everything. The span information existed in the dataset (the spans field is there and populated), but it was never used to compute the views that got stored.

So stored_large = 0 is not "there are no dominoes in this structure" — it's "there ARE dominoes but they were recorded as if they weren't."


recomp_large is the same idea but instead of reading the pre-stored field, it calls get_director_views_fn live, passing the actual spans from the dataset:

recomp_views = get_director_views_fn(target_structure, spans=target_spans)
Inside get_director_views_fn in structure_generator_v2.py, when a block ends with 'l' (large) AND spans are provided, it looks up whether that cell has a span partner that is also visible to that director. If yes, size = 2. If no, size = 1.

So recomp_large counts how many cells come back as size=2 when spans are actually used.

What the non-zero numbers mean

Structure 0 gets recomp_large = 8 — meaning 8 cells across D1/D2/D3 correctly show size=2. These are the domino cells that directors should have been describing as large blocks all along.

The difference between stored_large=0 and recomp_large=8 for structure 0 is exactly the information gap: 8 cells that existed as dominoes in the target were invisible to directors in the original runs. They saw unit blocks, not dominoes.



"""
import json
from collections import defaultdict
from structure_generator_v2 import get_director_views as get_director_views_fn

DATASET = "/home/hannah/CRAFT/CRAFT/data/structures_dataset_20.json"

with open(DATASET) as f:
    data = json.load(f)

def count_large(views):
    """Count cells with size=2 across all directors and rows."""
    return sum(
        1
        for d in views.values()
        for row in d.values()
        for cell in row
        if cell.get("size", 1) == 2
    )

def diff_views(stored, recomputed):
    """Return per-director count of cells where size differs."""
    diffs = {}
    for d in ("D1", "D2", "D3"):
        changed = 0
        for row in ("row_0", "row_1", "row_2"):
            s_row = stored.get(d, {}).get(row, [])
            r_row = recomputed.get(d, {}).get(row, [])
            for s, r in zip(s_row, r_row):
                if s.get("size") != r.get("size"):
                    changed += 1
        diffs[d] = changed
    return diffs

print(f"{'Struct':<8} {'Spans':>6} {'Stored large':>13} {'Recomp large':>13} {'D1 diff':>8} {'D2 diff':>8} {'D3 diff':>8}")
print("-" * 70)

for i, sample in enumerate(data):
    target_spans = {int(k): v for k, v in sample["spans"].items()}
    target_structure = sample["structure"]

    # Way 1: pre-stored (original paper)
    stored_views = sample["director_views"]

    # Way 2: live recomputed with spans (rebuttals fix)
    recomp_views = get_director_views_fn(target_structure, spans=target_spans)

    n_spans      = sum(len(v) for v in target_spans.values())
    stored_large = count_large(stored_views)
    recomp_large = count_large(recomp_views)
    diffs        = diff_views(stored_views, recomp_views)

    print(f"{i:<8} {n_spans:>6} {stored_large:>13} {recomp_large:>13} "
          f"{diffs['D1']:>8} {diffs['D2']:>8} {diffs['D3']:>8}")

print()
print("Stored large = cells with size=2 in dataset (should be 0 if bug present)")
print("Recomp large = cells with size=2 after span fix")
print("D1/D2/D3 diff = cells where size changed between the two versions")
