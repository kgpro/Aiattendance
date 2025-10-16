import os, shutil

def flatten_dataset(root):
    for split in ["train", "val"]:
        img_root = os.path.join(root, "images", split)
        lbl_root = os.path.join(root, "labels", split)

        # Create flat dirs
        flat_img = os.path.join(root, "images", f"{split}_flat")
        flat_lbl = os.path.join(root, "labels", f"{split}_flat")
        os.makedirs(flat_img, exist_ok=True)
        os.makedirs(flat_lbl, exist_ok=True)

        # Move files from subdirs
        for subdir, _, files in os.walk(img_root):
            for f in files:
                if f.endswith(".jpg"):
                    shutil.move(os.path.join(subdir, f), os.path.join(flat_img, f))

        for subdir, _, files in os.walk(lbl_root):
            for f in files:
                if f.endswith(".txt"):
                    shutil.move(os.path.join(subdir, f), os.path.join(flat_lbl, f))

        print(f"[OK] Flattened {split}: {len(os.listdir(flat_img))} images")

# Run
flatten_dataset("./dataset/widerface")
