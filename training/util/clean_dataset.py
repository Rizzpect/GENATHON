from pathlib import Path
import shutil

DATA = Path(__file__).resolve().parents[2] / "data"

def clean_split(split):
    imgs = (DATA / split / "images")
    lbls = (DATA / split / "labels")
    keep = 0
    drop = 0
    (imgs / "unlabeled").mkdir(exist_ok=True)
    for img in imgs.glob("*.*"):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}: 
            continue
        lab = lbls / (img.stem + ".txt")
        if lab.exists() and lab.stat().st_size > 0:
            keep += 1
        else:
            # move unlabeled images aside so training is clean
            shutil.move(str(img), str(imgs / "unlabeled" / img.name))
            drop += 1
    print(f"{split}: kept {keep}, moved {drop} unlabeled images → {imgs/'unlabeled'}")

for s in ["train", "valid", "test"]:
    clean_split(s)