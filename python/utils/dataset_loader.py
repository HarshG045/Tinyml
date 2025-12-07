# python/utils/dataset_loader.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def load_image_folder_dataset(
    root_dir,
    img_size=(64, 64),
    grayscale=True,
    test_split=0.2,
):
    """
    Load dataset from:
        root_dir/
            class0/
                img1.jpg
                img2.jpeg
                ...
            class1/
                ...
    Returns:
        (x_train, y_train), (x_val, y_val), num_classes, class_names
    """

    # allowed image extensions
    ALLOWED_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

    # ----------------------------
    # 1. Find classes
    # ----------------------------
    class_names = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )

    if not class_names:
        raise RuntimeError(
            f"No class folders found in dataset directory: {root_dir}"
        )

    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

    images = []
    labels = []

    # ----------------------------
    # 2. Load images from folders
    # ----------------------------
    for cls in class_names:
        cls_path = os.path.join(root_dir, cls)

        for fname in os.listdir(cls_path):
            fname_lower = fname.lower()

            # skip non-image files
            if not fname_lower.endswith(ALLOWED_EXT):
                continue

            fpath = os.path.join(cls_path, fname)

            img = cv2.imread(fpath)
            if img is None:
                print(f"[WARN] Could not read image: {fpath}")
                continue

            # convert to grayscale if needed
            if grayscale:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # resize
            img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

            # expand channels for grayscale
            if grayscale:
                img = np.expand_dims(img, axis=-1)

            # normalize to 0-1
            img = img.astype("float32") / 255.0

            images.append(img)
            labels.append(class_to_idx[cls])

    # ----------------------------
    # 3. Validate dataset
    # ----------------------------
    if len(images) == 0:
        raise RuntimeError(
            f"No images were loaded from dataset folder: {root_dir}. "
            f"Check file extensions or folder structure."
        )

    images = np.array(images, dtype="float32")
    labels = np.array(labels, dtype="int32")

    # ----------------------------
    # 4. Train / Validation Split
    # ----------------------------
    x_train, x_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=test_split,
        random_state=42,
        stratify=labels,
    )

    num_classes = len(class_names)
    return (x_train, y_train), (x_val, y_val), num_classes, class_names
