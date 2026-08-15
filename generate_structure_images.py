"""
Generate director-view PNG images for structures_dataset_20.json.
 

Output: one PNG per structure saved to OUT_DIR.
"""

import io
import os
import json
import base64
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# ── paths ─────────────────────────────────────────────────────────────────────
STRUCTURES_JSON = Path("data/structures_dataset_20.json")
OUT_DIR         = Path("data/craft_eval_structures_renders")

# ─────────────────────────────────────────
# PALETTE
# ─────────────────────────────────────────
BG_DARK     = (245, 245, 250)
BG_PANEL    = (255, 255, 255)
BG_CELL     = (230, 230, 240)
BORDER_DIM  = (180, 180, 200)
BORDER_MID  = (120, 120, 160)
TEXT_BRIGHT = (20,  20,  40)
TEXT_DIM    = (60,  60,  90)
TEXT_FAINT  = (120, 120, 150)

LAYER_ACCENT = {
    2: (60,  80,  200),
    1: (100, 120, 180),
    0: (140, 140, 170),
}

BLOCK_COLORS = {
    "yellow": ((245, 197,  24), (180, 140,   0)),
    "orange": ((255, 123,   0), (180,  80,   0)),
    "blue":   (( 37,  99, 235), ( 22,  60, 160)),
    "green":  (( 34, 197,  94), ( 10, 120,  50)),
    "red":    ((220,  38,  38), (150,  20,  20)),
    "none":   ((25,   25,  40), ( 35,  35,  55)),
}

COLOR_NAME_MAP = {
    "g": "green", "b": "blue", "r": "red",
    "y": "yellow", "o": "orange", "n": "none",
}

# ─────────────────────────────────────────
# LAYOUT CONSTANTS
# ─────────────────────────────────────────
CELL_W        = 52
CELL_H        = 42
CELL_GAP      = 5
LAYER_GAP     = 6
PANEL_PAD     = 16
PANEL_GAP     = 18
HEADER_H      = 44
LAYER_LABEL_W = 52

ROW_W         = 3 * CELL_W + 2 * CELL_GAP
PANEL_INNER_W = LAYER_LABEL_W + ROW_W
PANEL_W       = PANEL_INNER_W + 2 * PANEL_PAD
PANEL_INNER_H = 3 * CELL_H + 2 * LAYER_GAP
PANEL_H       = HEADER_H + PANEL_INNER_H + 2 * PANEL_PAD + 20

MINI_CELL = 58
MINI_GAP  = 6
MINI_W    = 3 * MINI_CELL + 2 * MINI_GAP + 2 * PANEL_PAD + 20
MINI_H    = PANEL_H + 20

IMG_W = 3 * PANEL_W + 2 * PANEL_GAP + PANEL_GAP + MINI_W + 2 * PANEL_PAD
IMG_H = PANEL_H + 80

DIRECTOR_META = {
    "D1": {"label": "Director 1", "desc": "Left col  (j=0)", "cells": ["(0,0)", "(1,0)", "(2,0)"]},
    "D2": {"label": "Director 2", "desc": "Top row  (i=0)",  "cells": ["(0,0)", "(0,1)", "(0,2)"]},
    "D3": {"label": "Director 3", "desc": "Right col  (j=2)","cells": ["(0,2)", "(1,2)", "(2,2)"]},
}

LAYER_LABEL_TEXT = {2: "L2 top", 1: "L1 mid", 0: "L0 base"}

# ─────────────────────────────────────────
# FONTS
# ─────────────────────────────────────────
def _font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()

def _font_reg(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    except Exception:
        return ImageFont.load_default()

FONT_TITLE = _font(15)
FONT_PANEL = _font(13)
FONT_LABEL = _font_reg(11)
FONT_TINY  = _font_reg(10)

# ─────────────────────────────────────────
# DRAWING PRIMITIVES
# ─────────────────────────────────────────
def _rounded_rect(draw, x0, y0, x1, y1, r, fill, outline=None, outline_width=1):
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r,
                           fill=fill, outline=outline, width=outline_width)


def _hatch(img, x0, y0, x1, y1):
    draw = ImageDraw.Draw(img)
    step = 8
    c1, c2 = (220, 220, 230), (200, 200, 215)
    draw.rectangle([x0, y0, x1, y1], fill=c1)
    for offset in range(-(y1 - y0), (x1 - x0), step):
        draw.line([(x0 + offset, y0), (x0 + offset + (y1 - y0), y1)], fill=c2, width=1)


def _draw_block(draw, img, x, y, color_name, is_large_left=False,
                is_large_right=False, is_large_solo=False):
    if is_large_right:
        return

    fill, border = BLOCK_COLORS.get(color_name, BLOCK_COLORS["none"])

    if color_name == "none":
        _hatch(img, x, y, x + CELL_W - 1, y + CELL_H - 1)
        draw.rectangle([x, y, x + CELL_W - 1, y + CELL_H - 1],
                       outline=BORDER_DIM, width=1)
        return

    w = (CELL_W * 2 + CELL_GAP) if is_large_left else CELL_W

    draw.rectangle([x + 2, y + 3, x + w + 1, y + CELL_H + 1], fill=(5, 5, 12))
    _rounded_rect(draw, x, y, x + w - 1, y + CELL_H - 1, r=5, fill=fill,
                  outline=border, outline_width=1)

    hi = tuple(min(255, c + 50) for c in fill)
    n_studs = 2 if is_large_left else 1
    stud_xs = [x + w // (n_studs + 1) * (i + 1) for i in range(n_studs)]
    for sx in stud_xs:
        draw.ellipse([sx - 4, y + CELL_H // 2 - 4, sx + 4, y + CELL_H // 2 + 4],
                     fill=hi, outline=border, width=1)

    label = color_name[:3].upper() + ("·L" if is_large_left else "·s")
    draw.text((x + 4, y + CELL_H - 13), label, font=FONT_TINY, fill=(220, 220, 255))


# ─────────────────────────────────────────
# DIRECTOR PANEL
# ─────────────────────────────────────────
def _draw_director_panel(draw, img, ox, oy, director_id, view_data, swap_labels=False):
    meta = DIRECTOR_META[director_id]

    _rounded_rect(draw, ox+3, oy+3, ox+PANEL_W+2, oy+PANEL_H+2, r=10, fill=(210, 210, 220))
    _rounded_rect(draw, ox, oy, ox + PANEL_W - 1, oy + PANEL_H - 1,
                  r=10, fill=BG_PANEL, outline=(200, 200, 215), outline_width=1)

    accent = (91, 94, 244)
    _rounded_rect(draw, ox + PANEL_PAD, oy + 10, ox + PANEL_PAD + 30, oy + 24, r=4, fill=accent)
    draw.text((ox + PANEL_PAD + 4, oy + 11), director_id, font=FONT_TINY, fill=(255, 255, 255))
    draw.text((ox + PANEL_PAD + 36, oy + 10), meta["label"], font=FONT_PANEL, fill=TEXT_BRIGHT)
    draw.text((ox + PANEL_PAD, oy + 26), meta["desc"], font=FONT_TINY, fill=TEXT_FAINT)
    draw.line([(ox + PANEL_PAD, oy + HEADER_H - 4), (ox + PANEL_W - PANEL_PAD, oy + HEADER_H - 4)],
              fill=BORDER_DIM, width=1)

    content_x = ox + PANEL_PAD + LAYER_LABEL_W
    content_y = oy + HEADER_H + PANEL_PAD

    for display_idx, layer_idx in enumerate([2, 1, 0]):
        row_key = f"row_{layer_idx}"
        cells = view_data[row_key]
        row_y = content_y + display_idx * (CELL_H + LAYER_GAP)

        draw.text((ox + PANEL_PAD, row_y + CELL_H // 2 - 5),
                  LAYER_LABEL_TEXT[layer_idx], font=FONT_TINY, fill=LAYER_ACCENT[layer_idx])
        _rounded_rect(draw, content_x - 4, row_y - 3,
                      content_x + ROW_W + 3, row_y + CELL_H + 3,
                      r=5, fill=(235, 237, 245), outline=None)

        flags = []
        i = 0
        while i < len(cells):
            c = cells[i]
            nxt = cells[i + 1] if i + 1 < len(cells) else None
            if (c["size"] == 2 and nxt and nxt["size"] == 2
                    and nxt["color"] == c["color"] and c["color"] != "none"):
                flags.append(("left", c["color"]))
                flags.append(("right", c["color"]))
                i += 2
            else:
                flags.append(("solo", c["color"]))
                i += 1
        while len(flags) < 3:
            flags.append(("solo", "none"))

        for ci, (role, color) in enumerate(flags):
            cx = content_x + ci * (CELL_W + CELL_GAP)
            _draw_block(draw, img, cx, row_y, color,
                        is_large_left=(role == "left"),
                        is_large_right=(role == "right"),
                        is_large_solo=(role == "solo" and ci < len(cells) and cells[ci]["size"] == 2))

    coord_y = content_y + 3 * CELL_H + 2 * LAYER_GAP + 2
    coords = list(reversed(meta["cells"])) if swap_labels else meta["cells"]
    for ci, coord in enumerate(coords):
        cx = content_x + ci * (CELL_W + CELL_GAP)
        draw.text((cx + CELL_W // 2 - 12, coord_y), coord, font=FONT_TINY, fill=TEXT_FAINT)


# ─────────────────────────────────────────
# FULL GRID MINIMAP
# ─────────────────────────────────────────
def _draw_minimap(draw, img, ox, oy, structure):
    _rounded_rect(draw, ox, oy, ox + MINI_W - 1, oy + MINI_H - 1,
                  r=10, fill=BG_PANEL, outline=BORDER_DIM, outline_width=1)
    draw.text((ox + PANEL_PAD, oy + 10), "FULL GRID", font=FONT_PANEL, fill=TEXT_DIM)
    draw.text((ox + PANEL_PAD, oy + 24), "top-block color · height", font=FONT_TINY, fill=TEXT_FAINT)

    grid_x = ox + PANEL_PAD
    grid_y = oy + HEADER_H
    rows = [
        ["(0,0)", "(0,1)", "(0,2)"],
        ["(1,0)", "(1,1)", "(1,2)"],
        ["(2,0)", "(2,1)", "(2,2)"],
    ]
    for ri, row in enumerate(rows):
        for ci, coord in enumerate(row):
            stack = structure.get(coord, [])
            height = len(stack)
            top_block = stack[-1] if stack else None
            color_key = COLOR_NAME_MAP.get(top_block[0], "none") if top_block else "none"
            fill, border = BLOCK_COLORS[color_key]

            cx = grid_x + ci * (MINI_CELL + MINI_GAP)
            cy = grid_y + ri * (MINI_CELL + MINI_GAP)

            if height == 0:
                _hatch(img, cx, cy, cx + MINI_CELL - 1, cy + MINI_CELL - 1)
                draw.rectangle([cx, cy, cx + MINI_CELL - 1, cy + MINI_CELL - 1],
                               fill=(220, 220, 230), outline=(180, 180, 200), width=1)
            else:
                _rounded_rect(draw, cx, cy, cx + MINI_CELL - 1, cy + MINI_CELL - 1,
                               r=5, fill=fill, outline=border, outline_width=2)
                draw.text((cx + 4, cy + 4), str(height), font=FONT_LABEL, fill=(20, 20, 40))

            draw.text((cx + 3, cy + MINI_CELL - 13), coord, font=FONT_TINY, fill=(80, 80, 110))


# ─────────────────────────────────────────
# VIEW COMPUTATION
# ─────────────────────────────────────────
_COLOR_NAMES = {"g": "green", "b": "blue", "r": "red",
                "y": "yellow", "o": "orange", "n": "none"}

def _compute_views(structure, spans):
    director_coords = {
        "D1": ["(0,0)", "(1,0)", "(2,0)"],
        "D2": ["(0,0)", "(0,1)", "(0,2)"],
        "D3": ["(0,2)", "(1,2)", "(2,2)"],
    }
    def cell(coord, layer, visible_coords):
        stack = structure.get(coord, [])
        if layer >= len(stack):
            return {"color": "none", "size": 1}
        block = stack[layer]
        color = _COLOR_NAMES.get(block[0], "none")
        if block.endswith("l"):
            layer_spans = spans.get(layer, [])
            partner = next(
                (b if a == coord else a for a, b in layer_spans if coord in (a, b)),
                None
            )
            size = 2 if (partner and partner in visible_coords) else 1
        else:
            size = 1
        return {"color": color, "size": size}

    views = {}
    for did, coords in director_coords.items():
        views[did] = {f"row_{l}": [cell(c, l, coords) for c in coords] for l in range(3)}
    return views


# ─────────────────────────────────────────
# TOP-LEVEL RENDER FUNCTION
# ─────────────────────────────────────────
def render_structure_views(structure, spans, views=None, structure_id="structure",
                           partial=None, return_bytes=True, swap_d2d3=False):
    int_spans = {int(k): [tuple(pair) for pair in v] for k, v in spans.items()}

    if views is None:
        views = _compute_views(structure, int_spans)

    if swap_d2d3:
        views = {
            d: ({row: cells[::-1] for row, cells in rows.items()} if d in ("D2", "D3") else rows)
            for d, rows in views.items()
        }

    img  = Image.new("RGB", (IMG_W, IMG_H), BG_DARK)
    draw = ImageDraw.Draw(img)

    draw.text((PANEL_PAD, 12), structure_id, font=FONT_TITLE, fill=TEXT_BRIGHT)
    draw.text((PANEL_PAD, 30), "CRAFT · Director Perspective Views", font=FONT_TINY, fill=TEXT_FAINT)

    panel_y = 56
    for pi, did in enumerate(["D1", "D2", "D3"]):
        panel_x = PANEL_PAD + pi * (PANEL_W + PANEL_GAP)
        _draw_director_panel(draw, img, panel_x, panel_y, did, views[did],
                             swap_labels=(swap_d2d3 and did in ("D2", "D3")))

    mini_x = PANEL_PAD + 3 * (PANEL_W + PANEL_GAP)
    _draw_minimap(draw, img, mini_x, panel_y, partial if partial is not None else structure)

    legend_y = panel_y + PANEL_H + 10
    lx = PANEL_PAD
    draw.text((lx, legend_y), "COLORS:", font=FONT_TINY, fill=TEXT_FAINT)
    lx += 56
    for name, (fill, border) in BLOCK_COLORS.items():
        if name == "none":
            continue
        _rounded_rect(draw, lx, legend_y, lx + 12, legend_y + 12,
                      r=2, fill=fill, outline=border, outline_width=1)
        draw.text((lx + 15, legend_y + 1), name, font=FONT_TINY, fill=TEXT_DIM)
        lx += 15 + len(name) * 7 + 12

    if return_bytes:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    return img


# ─────────────────────────────────────────
# GENERATE ALL IMAGES
# ─────────────────────────────────────────
if __name__ == "__main__":
    with open(STRUCTURES_JSON) as f:
        dataset = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sample in dataset:
        sid       = sample["id"]
        structure = sample["structure"]
        spans     = {int(k): [tuple(pair) for pair in v] for k, v in sample["spans"].items()}

        png = render_structure_views(
            structure, spans,
            structure_id=sid,
            swap_d2d3=True,
        )

        out_path = OUT_DIR / f"{sid}.png"
        out_path.write_bytes(png)
        print(f"rendered {sid}")

    print(f"\nDone — {len(dataset)} images in {OUT_DIR}/")
