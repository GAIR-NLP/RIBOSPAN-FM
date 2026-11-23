#!/usr/bin/env python3
"""Extract the paper figures from the manuscript PDF into assets/.

Each figure is cropped to the content above its caption, rendered at a high
DPI and downsampled to README width.

    python scripts/extract_paper_figures.py path/to/manuscript.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import pymupdf
from PIL import Image

ASSETS = Path(__file__).resolve().parent.parent / "assets"
RENDER_DPI = 400
MAX_WIDTH = 1800

# 0-based page index -> (caption prefix, output stem)
FIGURES = {
    2:  ("Fig. 1", "fig1_overview"),
    3:  ("Fig. 2", "fig2_zeroshot"),
    6:  ("Fig. 3", "fig3_architecture"),
    7:  ("Fig. 4", "fig4_layerwise"),
    8:  ("Fig. 5", "fig5_secondary_structure"),
    10: ("Fig. 6", "fig6_downstream"),
    23: ("Extended Data Fig. 2", "edfig2_model_scaling"),
}


def figure_clip(page, caption: str):
    """Bounding box of everything on the page above the caption."""
    hits = page.search_for(caption)
    if not hits:
        return None
    top = min(h.y0 for h in hits)
    rects = [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")]
    rects += [d["rect"] for d in page.get_drawings()]
    rects += [page.get_image_bbox(i) for i in page.get_images(full=True)]
    rects = [r for r in rects if r.y1 < top - 2]
    if not rects:
        return None
    m = 6
    return pymupdf.Rect(max(0, min(r.x0 for r in rects) - m),
                        max(0, min(r.y0 for r in rects) - m),
                        min(page.rect.x1, max(r.x1 for r in rects) + m),
                        top - m)


def main(pdf_path: str):
    doc = pymupdf.open(pdf_path)
    ASSETS.mkdir(exist_ok=True)
    for page_idx, (caption, stem) in FIGURES.items():
        page = doc[page_idx]
        clip = figure_clip(page, caption)
        if clip is None:
            print(f"[skip] {stem}: caption {caption!r} not found on page {page_idx + 1}")
            continue
        pix = page.get_pixmap(dpi=RENDER_DPI, clip=clip)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        if img.width > MAX_WIDTH:
            h = round(img.height * MAX_WIDTH / img.width)
            img = img.resize((MAX_WIDTH, h), Image.LANCZOS)
        out = ASSETS / f"{stem}.png"
        img.save(out, optimize=True)
        print(f"{stem:26s} page {page_idx + 1:2d}  {img.width}x{img.height}  "
              f"{out.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "manuscript.pdf")
