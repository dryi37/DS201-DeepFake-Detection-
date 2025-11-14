from glob import glob
import os
import json
from tqdm import tqdm

DATASET_DIRS = ["Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
FAKE_DIRS = {"Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"}


def init_ff(data_root="data", phase="train"):
    split_path = os.path.join(data_root, "FaceForensics++", f"{phase}.json")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Không thấy file split: {split_path}")

    # Flatten list các ID
    sample_ids = sum(json.load(open(split_path, "r")), [])

    video_dirs, labels = [], []

    for sid in tqdm(sample_ids, desc=f"Indexing {phase}"):
        # Lấy ảnh REAL
        orig_dir = os.path.join(data_root, "Original", "faces", sid)
        if os.path.isdir(orig_dir) and len(glob(os.path.join(orig_dir, "*.png"))) > 0:
            video_dirs.append(orig_dir)
            labels.append(0)

        # Lấy ảnh FAKE từ tất cả các loại fake
        for fake_ds in FAKE_DIRS:
            fake_root = os.path.join(data_root, fake_ds, "faces")
            if not os.path.exists(fake_root):
                continue

            fake_dirs = glob(os.path.join(fake_root, f"{sid}_*"))
            for fd in fake_dirs:
                if os.path.isdir(fd) and len(glob(os.path.join(fd, "*.png"))) > 0:
                    video_dirs.append(fd)
                    labels.append(1)

    print(f"[INFO] Tổng số video: {len(video_dirs)} (phase={phase})")
    print(f"[INFO] Real: {labels.count(0)} | Fake: {labels.count(1)}")
    return video_dirs, labels


if __name__ == "__main__":
    vids, labels = init_ff(data_root="data", phase="train")
    print("Ví dụ:", vids[:3], labels[:3])