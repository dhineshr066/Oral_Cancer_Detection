"""
generate_normal_masks.py
========================
Utility: Automatically generates all-black mask images for
every image in  dataset/normal/images/
and saves them to  dataset/normal/masks/

Run once after adding your normal images:
  python generate_normal_masks.py
"""

import os
import cv2
import numpy as np

BASE_DIR  = os.path.dirname(__file__)
IMG_DIR   = os.path.join(BASE_DIR, 'dataset', 'normal', 'images')
MASK_DIR  = os.path.join(BASE_DIR, 'dataset', 'normal', 'masks')

os.makedirs(MASK_DIR, exist_ok=True)

count = 0
for fname in sorted(os.listdir(IMG_DIR)):
    ext = os.path.splitext(fname)[1].lower()
    if ext not in ('.jpg', '.jpeg', '.png'):
        continue

    img = cv2.imread(os.path.join(IMG_DIR, fname))
    if img is None:
        print(f"[WARN] Cannot read {fname}")
        continue

    h, w = img.shape[:2]
    black_mask = np.zeros((h, w), dtype=np.uint8)

    stem      = os.path.splitext(fname)[0]
    mask_path = os.path.join(MASK_DIR, stem + '.png')
    cv2.imwrite(mask_path, black_mask)
    count += 1
    print(f"  Created mask: {stem}.png  ({w}x{h})")

print(f"\n[INFO] Done. {count} black masks created in {MASK_DIR}")
