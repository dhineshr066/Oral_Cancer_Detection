"""
train.py
========
Training Pipeline — Oral Cancer Detection (U-Net Segmentation)

Dataset structure expected:
  dataset/
    cancer/
      images/   <- cancer oral photos (.jpg / .png)
      masks/    <- binary masks  (white=tumor, black=bg)
    normal/
      images/   <- normal oral photos
      masks/    <- all-black masks

Usage:
  python train.py --use_resnet True  --epochs 30 --batch_size 8
  python train.py --use_resnet False --epochs 50 --batch_size 8
"""

import os, sys, argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import albumentations as A

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))
from unet_model import get_model, dice_coefficient, iou_metric

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
IMG_SIZE        = 224
BATCH_SIZE      = 8
EPOCHS          = 30
SEED            = 42
BASE_DIR        = os.path.dirname(__file__)
DATASET_DIR     = os.path.join(BASE_DIR, 'dataset')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'model', 'best_model.h5')

# ─────────────────────────────────────────────
# Augmentation Pipelines
# ─────────────────────────────────────────────
train_aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), p=0.4),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),
    A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3, p=0.2),
    A.Resize(IMG_SIZE, IMG_SIZE),
], additional_targets={'mask': 'mask'})

val_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
], additional_targets={'mask': 'mask'})


# ─────────────────────────────────────────────
# Dataset Loading  (NEW STRUCTURE)
# ─────────────────────────────────────────────
def load_dataset(dataset_dir=DATASET_DIR):
    """
    Walks  dataset/cancer/images  and  dataset/normal/images.
    Matches each image to its mask in  .../masks/
    Returns img_paths, mask_paths, labels (1=cancer, 0=normal)
    """
    img_paths, mask_paths, labels = [], [], []

    for category, label in [('cancer', 1), ('normal', 0)]:
        img_dir  = os.path.join(dataset_dir, category, 'images')
        mask_dir = os.path.join(dataset_dir, category, 'masks')

        if not os.path.isdir(img_dir):
            print(f"[WARN] Not found: {img_dir}")
            continue

        for fname in sorted(os.listdir(img_dir)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ('.jpg', '.jpeg', '.png'):
                continue

            img_path  = os.path.join(img_dir, fname)
            stem      = os.path.splitext(fname)[0]
            mask_path = os.path.join(mask_dir, stem + '.png')
            if not os.path.exists(mask_path):
                mask_path = os.path.join(mask_dir, fname)
            if not os.path.exists(mask_path):
                print(f"[WARN] No mask for {fname} — skipping.")
                continue

            img_paths.append(img_path)
            mask_paths.append(mask_path)
            labels.append(label)

    print(f"[INFO] Samples loaded: {len(img_paths)}  "
          f"(cancer={labels.count(1)}, normal={labels.count(0)})")
    return img_paths, mask_paths, labels


# ─────────────────────────────────────────────
# Dataset Class
# ─────────────────────────────────────────────
class SegDataset:
    def __init__(self, img_paths, mask_paths, aug=None):
        self.img_paths  = img_paths
        self.mask_paths = mask_paths
        self.aug        = aug

    def __len__(self): return len(self.img_paths)

    def __getitem__(self, idx):
        img  = cv2.cvtColor(cv2.imread(self.img_paths[idx]),  cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,  (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        if self.aug:
            out  = self.aug(image=img, mask=mask)
            img, mask = out['image'], out['mask']

        img  = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)[..., np.newaxis]
        return img, mask

    def to_tf_dataset(self, batch_size, shuffle=True):
        def gen():
            for i in range(len(self)):
                yield self[i]
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
            )
        )
        if shuffle:
            ds = ds.shuffle(len(self), seed=SEED)
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ─────────────────────────────────────────────
# Callbacks
# ─────────────────────────────────────────────
def get_callbacks():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    return [
        ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_dice_coefficient',
                        mode='max', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_dice_coefficient', mode='max',
                      patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=5, min_lr=1e-7, verbose=1),
        TensorBoard(log_dir=os.path.join(BASE_DIR, 'logs')),
    ]


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────
def plot_history(history):
    keys = [
        ('accuracy',         'Accuracy'),
        ('loss',             'Loss'),
        ('dice_coefficient', 'Dice Coefficient'),
        ('iou_metric',       'IoU'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Training History – Oral Cancer Segmentation',
                 fontsize=14, fontweight='bold')
    for ax, (k, title) in zip(axes.flat, keys):
        if k in history.history:
            ax.plot(history.history[k],          label='Train', lw=2)
            ax.plot(history.history[f'val_{k}'], label='Val',   lw=2, ls='--')
        ax.set_title(title); ax.set_xlabel('Epoch')
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'training_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved -> {out}")
    plt.show()


def visualize_preds(model, tf_ds, n=4):
    batch_imgs, batch_masks = next(iter(tf_ds.unbatch().batch(n)))
    preds = model.predict(batch_imgs, verbose=0)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    for col, t in enumerate(['Original', 'Ground Truth', 'Prediction']):
        axes[0][col].set_title(t, fontsize=12, fontweight='bold')
    for i in range(n):
        axes[i][0].imshow(batch_imgs[i])
        axes[i][1].imshow(batch_masks[i,:,:,0], cmap='gray')
        axes[i][2].imshow(preds[i,:,:,0],       cmap='hot')
        for col in range(3): axes[i][col].axis('off')
    plt.tight_layout()
    out = os.path.join(BASE_DIR, 'sample_predictions.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"[INFO] Saved -> {out}")
    plt.show()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(use_resnet=True, epochs=EPOCHS, batch_size=BATCH_SIZE):
    print("=" * 58)
    print("  Oral Cancer Detection  --  U-Net Segmentation")
    print("=" * 58)

    img_paths, mask_paths, _ = load_dataset()
    if not img_paths:
        sys.exit("[ERROR] No data found. Add images to dataset/ folders.")

    tr_imgs, vl_imgs, tr_masks, vl_masks = train_test_split(
        img_paths, mask_paths, test_size=0.2, random_state=SEED
    )
    print(f"[INFO] Train={len(tr_imgs)}  Val={len(vl_imgs)}")

    train_ds = SegDataset(tr_imgs, tr_masks, aug=train_aug).to_tf_dataset(batch_size)
    val_ds   = SegDataset(vl_imgs, vl_masks, aug=val_aug).to_tf_dataset(batch_size, shuffle=False)

    model = get_model(use_resnet=use_resnet, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    model.summary()

    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=epochs, callbacks=get_callbacks()
    )

    print("\n[INFO] Final Validation Metrics:")
    for name, val in zip(model.metrics_names, model.evaluate(val_ds, verbose=1)):
        print(f"  {name}: {val:.4f}")

    plot_history(history)
    visualize_preds(model, val_ds, n=min(4, len(vl_imgs)))
    print(f"\n[INFO] Best model -> {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--use_resnet', type=lambda x: x.lower()=='true', default=True)
    ap.add_argument('--epochs',     type=int, default=EPOCHS)
    ap.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    args = ap.parse_args()
    main(args.use_resnet, args.epochs, args.batch_size)
