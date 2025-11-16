from glob import glob
import os
import json

def faceforensics(dataset='all', phase='test'):
    valid_sets = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

    if isinstance(dataset, str):
        if dataset == 'all':
            selected_sets = valid_sets
        else:
            assert dataset in valid_sets, f"Dataset '{dataset}' không hợp lệ!"
            selected_sets = [dataset]

    elif isinstance(dataset, list):
        for d in dataset:
            assert d in valid_sets, f"Dataset '{d}' không hợp lệ!"
        selected_sets = dataset

    else:
        raise TypeError("dataset phải là str hoặc list")

    base_dir = "data"
    real_path = os.path.join(base_dir, "Original", "faces")   # FIXED
    ffpp_json = os.path.join(base_dir, "FaceForensics++", f"{phase}.json")

    if not os.path.exists(ffpp_json):
        raise FileNotFoundError(f"[ERROR] Không tìm thấy file JSON: {ffpp_json}")

    list_dict = json.load(open(ffpp_json, "r"))
    prefix_list = [i for group in list_dict for i in group]

    folder_list = []
    label_list = []

    real_folders = sorted(glob(os.path.join(real_path, "*")))
    real_folders = [
        f for f in real_folders
        if os.path.basename(f)[:3] in prefix_list
    ]

    folder_list += real_folders
    label_list += [0] * len(real_folders)

    for fake_name in selected_sets:
        fake_path = os.path.join(base_dir, fake_name, "faces")   # FIXED
        fake_folders = sorted(glob(os.path.join(fake_path, "*")))

        fake_folders = [
            f for f in fake_folders
            if os.path.basename(f)[:3] in prefix_list
        ]

        folder_list += fake_folders
        label_list += [1] * len(fake_folders)

    fakes = sum(label_list)
    reals = len(label_list) - fakes

    print(f"[INFO] Loaded {len(folder_list)} videos ({fakes} fakes, {reals} reals)")
    return folder_list, label_list
