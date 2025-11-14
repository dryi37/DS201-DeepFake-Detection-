import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import torch
from retinaface.pre_trained_models import get_model
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def expand_bbox(x0, y0, x1, y1, img_w, img_h, margin_ratio=0.1):
    bw, bh = x1 - x0, y1 - y0
    dw, dh = int(bw * margin_ratio), int(bh * margin_ratio)
    x0 = max(0, x0 - dw)
    y0 = max(0, y0 - dh)
    x1 = min(img_w, x1 + dw)
    y1 = min(img_h, y1 + dh)
    return x0, y0, x1, y1


def extract_faces(model, video_path, save_root, num_frames=16, face_size=224, margin_ratio=0.1):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_dir = os.path.join(save_root, "faces", video_name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        tqdm.write(f"[ERROR] Cannot open video: {video_name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        tqdm.write(f"[ERROR] Empty video: {video_name}")
        cap.release()
        return

    frame_idxs = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    skip_video = False

    for i, frame_idx in enumerate(frame_idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            tqdm.write(f"[WARN] Cannot read frame {frame_idx} in {video_name}")
            skip_video = True
            break

        img_h, img_w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            faces = model.predict_jsons(rgb)
        except Exception as e:
            tqdm.write(f"[ERROR] RetinaFace failed in {video_name} frame {i}: {e}")
            skip_video = True
            break

        # Không có mặt hoặc bbox lỗi
        if not faces:
            tqdm.write(f"[INFO] No face found in {video_name} frame {i}")
            skip_video = True
            break

        # Lọc bbox hợp lệ
        bboxes = [f["bbox"] for f in faces if "bbox" in f and len(f["bbox"]) == 4]
        if len(bboxes) == 0:
            tqdm.write(f"[WARN] Invalid bbox in {video_name} frame {i}")
            skip_video = True
            break

        # Lấy bbox lớn nhất
        sizes = [(x2 - x1) * (y2 - y1) for (x1, y1, x2, y2) in bboxes]
        x0, y0, x1, y1 = map(int, bboxes[np.argmax(sizes)])
        x0, y0, x1, y1 = expand_bbox(x0, y0, x1, y1, img_w, img_h, margin_ratio)

        face_crop = frame[y0:y1, x0:x1]
        if face_crop.size == 0:
            tqdm.write(f"[WARN] Empty crop in {video_name} frame {i}")
            skip_video = True
            break

        face_crop = cv2.resize(face_crop, (face_size, face_size))
        cv2.imwrite(os.path.join(save_dir, f"{i:03d}.png"), face_crop)

    cap.release()
    # torch.cuda.empty_cache()

    if skip_video:
        tqdm.write(f"[SKIP] Video {video_name} skipped due to frame error.")
        try:
            for f in glob(os.path.join(save_dir, "*.png")):
                os.remove(f)
            os.rmdir(save_dir)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Extract 224x224 faces, skip video if any frame error.")
    parser.add_argument("-d", "--dataset", required=True,
                        choices=["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "Original"],
                        help="Tên dataset trong thư mục data/")
    parser.add_argument("-n", "--num_frames", type=int, default=16,
                        help="Số frame cần trích từ mỗi video")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị chạy RetinaFace (cpu/cuda)")
    parser.add_argument("--face_size", type=int, default=224,
                        help="Kích thước ảnh khuôn mặt sau khi resize")
    parser.add_argument("--margin", type=float, default=0.1,
                        help="Tỉ lệ mở rộng bbox quanh mặt (mặc định: 0.1 = 10%)")
    args = parser.parse_args()

    dataset_root = os.path.join("data", args.dataset)
    video_paths = sorted(glob(os.path.join(dataset_root, "*.mp4")))
    if len(video_paths) == 0:
        print(f"[ERROR] Không tìm thấy video nào trong {dataset_root}")
        return

    print(f"[INFO] {len(video_paths)} videos found in {args.dataset}")
    model = get_model("resnet50_2020-07-20", max_size=2048, device=args.device)
    model.eval()

    for idx, video_path in enumerate(tqdm(video_paths, desc=f"Extracting faces from {args.dataset}")):
        extract_faces(model, video_path, dataset_root,
                      num_frames=args.num_frames,
                      face_size=args.face_size,
                      margin_ratio=args.margin)

        if (idx + 1) % 200 == 0:
            torch.cuda.empty_cache()

    print("\n[INFO] Extraction completed.")

if __name__ == "__main__":
    main()