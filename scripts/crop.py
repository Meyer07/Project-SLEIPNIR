"""Cut per-position crops from a full overhead image (640x480).

Each crop is centred on ONE numbered target position, so the classifier label
describes that position even though neighbouring vials are visible. Two scales
are written for every position:
  tight/    small crop (default 72 px): cap and vial body of the target
  context/  wider crop (default 144 px): neighbours, for damage and for telling
            a transparent vial body from an empty fixture hole

Usage:
    python crops.py image.png --positions twin_positions_px.csv --fixture G --out crops/
    python crops.py path/to/images/ --positions twin_positions_px.csv --fixture G --out crops/   # whole folder
    (add --limit 50 to try the first 50 frames; macOS '._*' files are skipped)

The positions CSV needs the columns: fix, pos, u_px, v_px (pixel centre of each
position in the 640x480 image). twin_positions_px.csv has them for fixture G;
its S rows are NOT aperture centres (see the 'note' column), so use the official
fixture coordinates for S when you have them.

Output: crops/tight/*.png, crops/context/*.png and crops/index.csv, which maps
every crop back to its source image, fixture and position.
"""
import argparse
import csv
from pathlib import Path

import cv2


def load_positions(path, fixture=None):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    need = {"fix", "pos", "u_px", "v_px"}
    if not rows or not need <= set(rows[0]):
        raise ValueError(f"{path} must have columns {sorted(need)}")
    if fixture:
        rows = [r for r in rows if r["fix"] == fixture]
    return rows


def crop_at(img, u, v, size, pad_img=None):
    """size x size crop centred on pixel (u, v). Edges are reflected, so a
    position near the image border still gives a full-size crop."""
    half = size // 2
    p = size  # padding on each side; must be < image size
    if pad_img is None:
        pad_img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_REFLECT)
    cx, cy = int(round(float(u))) + p, int(round(float(v))) + p
    return pad_img[cy - half: cy - half + size, cx - half: cx - half + size]


def make_crops(image_path, positions, out_dir, tight=72, context=144):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    out = Path(out_dir)
    (out / "tight").mkdir(parents=True, exist_ok=True)
    (out / "context").mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    pad_t = cv2.copyMakeBorder(img, tight, tight, tight, tight, cv2.BORDER_REFLECT)
    pad_c = cv2.copyMakeBorder(img, context, context, context, context, cv2.BORDER_REFLECT)
    index = []
    for r in positions:
        name = f"{stem}__{r['fix']}{int(r['pos']):02d}.png"
        cv2.imwrite(str(out / "tight" / name), crop_at(img, r["u_px"], r["v_px"], tight, pad_t))
        cv2.imwrite(str(out / "context" / name), crop_at(img, r["u_px"], r["v_px"], context, pad_c))
        index.append({"crop": name, "image": Path(image_path).name, "fixture": r["fix"],
                      "position": int(r["pos"]), "u_px": r["u_px"], "v_px": r["v_px"],
                      "label": ""})  # fill in with pass/missing_cap/crooked_cap/damaged_vial/no_vial
    new = not (out / "index.csv").exists()
    with open(out / "index.csv", "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        if new:
            w.writeheader()
        w.writerows(index)
    return index


def list_images(path, limit=None):
    p = Path(path)
    if p.is_dir():
        imgs = sorted(x for x in p.iterdir()
                      if x.suffix.lower() in (".jpg", ".jpeg", ".png") and not x.name.startswith("."))
    else:
        imgs = [p]
    return imgs[:limit] if limit else imgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="an image file, or a folder of images")
    ap.add_argument("--positions", required=True)
    ap.add_argument("--fixture", choices=["G", "S"], default=None, help="default: both")
    ap.add_argument("--out", default="crops")
    ap.add_argument("--tight", type=int, default=72)
    ap.add_argument("--context", type=int, default=144)
    ap.add_argument("--limit", type=int, default=None, help="only the first N images of a folder")
    a = ap.parse_args()
    pos = load_positions(a.positions, a.fixture)
    if any("NOT aperture" in (r.get("note") or "") for r in pos):
        print("WARNING: some selected rows are not true aperture centres (see 'note'). "
              "Use the official fixture coordinates for those.")
    imgs = list_images(a.image, a.limit)
    if not imgs:
        raise SystemExit(f"no images found in {a.image}")
    total = 0
    for i, im in enumerate(imgs, 1):
        total += len(make_crops(im, pos, a.out, a.tight, a.context))
        if len(imgs) > 1 and i % 100 == 0:
            print(f"  {i}/{len(imgs)} images")
    print(f"wrote {total} crop pairs from {len(imgs)} image(s) to {a.out}/ "
          f"(tight={a.tight}px, context={a.context}px)")


if __name__ == "__main__":
    main()