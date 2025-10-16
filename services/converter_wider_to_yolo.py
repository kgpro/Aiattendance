import os
from PIL import Image
from tqdm import tqdm

# ---- Edit these if your paths differ ----
DATA_ROOT = "dataset/widerface"
TRAIN_SPLIT_TXT = os.path.join(DATA_ROOT, "wider_face_train_bbx_gt.txt")
VAL_SPLIT_TXT   = os.path.join(DATA_ROOT, "wider_face_val_bbx_gt.txt")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "images/train")
VAL_IMG_DIR   = os.path.join(DATA_ROOT, "images/val")

TRAIN_LBL_DIR = os.path.join(DATA_ROOT, "labels/train")
VAL_LBL_DIR   = os.path.join(DATA_ROOT,  "labels/val")
# -----------------------------------------


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def clamp01(x):
    return max(0.0, min(1.0, x))

def convert_split(split_txt, images_dir, labels_dir):
    ensure_dir(labels_dir)

    with open(split_txt, "r") as f:
        lines = [ln.strip() for ln in f.readlines()]

    i = 0
    num_images = 0
    num_boxes  = 0
    missing = 0

    pbar = tqdm(total=len(lines), desc=f"Converting {os.path.basename(split_txt)}")
    while i < len(lines):
        rel_path = lines[i]; i += 1
        pbar.update(1)

        try:# number of face boxes
            n = int(lines[i]); i += 1
            pbar.update(1)
        except Exception as e:
            print("corrupted line found jumping... /n",e)
            continue

        # image + label paths
        img_path = os.path.join(images_dir, rel_path)
        lbl_path = os.path.join(labels_dir, rel_path.replace(".jpg", ".txt"))
        os.makedirs(os.path.dirname(lbl_path), exist_ok=True)

        # load image size
        try:
            with Image.open(img_path) as img:
                w, h = img.size
        except Exception:
            # image missing or unreadable
            missing += 1
            # still need to skip the n lines describing boxes
            for _ in range(n):
                i += 1
                pbar.update(1)
            # and create an empty label file to keep index aligned
            open(lbl_path, "w").close()
            continue

        # write labels
        written = 0
        with open(lbl_path, "w") as out:
            for _ in range(n):
                parts = lines[i].split()
                i += 1
                pbar.update(1)

                # x y w h are the first 4 numbers
                x, y, bw, bh = map(float, parts[:4])

                # "invalid" flag is the 8th (0-based index 7)
                invalid = int(parts[7]) if len(parts) > 7 else 0
                if invalid == 1:
                    continue

                if bw <= 0 or bh <= 0 or w <= 0 or h <= 0:
                    continue

                xc = (x + bw / 2.0) / w
                yc = (y + bh / 2.0) / h
                ww = bw / w
                hh = bh / h

                # clamp to [0,1]
                xc = clamp01(xc); yc = clamp01(yc); ww = clamp01(ww); hh = clamp01(hh)

                # single class "face" -> class id 0
                out.write(f"0 {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")
                written += 1
                num_boxes += 1

        if written == 0:
            # ensure empty file exists
            open(lbl_path, "a").close()

        num_images += 1

    pbar.close()
    print(f"[{os.path.basename(split_txt)}] Images: {num_images}, Boxes kept: {num_boxes}, Missing: {missing}")


if __name__ == "__main__":
    # You can copy the two TXT files into DATA_ROOT, or point TRAIN_SPLIT_TXT/VAL_SPLIT_TXT to wider_face_split/ paths.
    assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
    assert os.path.isdir(VAL_IMG_DIR),   f"Missing: {VAL_IMG_DIR}"

    # If your TXT files are still in 'wider_face_split/', set these accordingly:
    if not os.path.isfile(TRAIN_SPLIT_TXT):
        TRAIN_SPLIT_TXT = "wider_face_split/wider_face_train_bbx_gt.txt"
    if not os.path.isfile(VAL_SPLIT_TXT):
        VAL_SPLIT_TXT = "wider_face_split/wider_face_val_bbx_gt.txt"

    convert_split(TRAIN_SPLIT_TXT, TRAIN_IMG_DIR, TRAIN_LBL_DIR)
    convert_split(VAL_SPLIT_TXT,   VAL_IMG_DIR,   VAL_LBL_DIR)
    print("Done. YOLO labels created.")
