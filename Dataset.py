import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

PATCH_SIZE  = 256
IMG_FULL_H  = 1400
IMG_FULL_W  = 2100
N_ROWS      = IMG_FULL_H // PATCH_SIZE   # 5
N_COLS      = IMG_FULL_W // PATCH_SIZE   # 8
N_PATCHES   = N_ROWS * N_COLS            # 40

IMG_SIZE    = (PATCH_SIZE, PATCH_SIZE)
BATCH_SIZE  = 8
NUM_CLASSES = 4
CLASSES     = ['Fish', 'Flower', 'Gravel', 'Sugar']


# --- 1. Parse CSV ---
def parse_csv(csv_path):
    train_df = pd.read_csv(csv_path)

    train_df[['ImageId', 'Label']] = train_df['Image_Label'].str.split('_', expand=True)
    train_df = train_df.drop(columns=['Image_Label'])
    train_df['EncodedPixels'] = train_df['EncodedPixels'].fillna('')
    train_df = train_df.pivot(index='ImageId', columns='Label', values='EncodedPixels').reset_index()
    train_df.fillna('', inplace=True)

    train_df['fold'] = -1
    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        train_df.loc[val_idx, 'fold'] = fold

    train_data = train_df[train_df['fold'] != 0].reset_index(drop=True)
    val_data   = train_df[train_df['fold'] == 0].reset_index(drop=True)
    return train_data, val_data


# --- 2. RLE decode ---
def rle2mask(rle, shape=(1400, 2100)):
    if rle == '':
        return np.zeros(shape, dtype=np.float32)
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')


# --- 3. Extract non-overlapping 256x256 patches from full-res image and mask ---
def _extract_patches(img, mask):
    crop_h = N_ROWS * PATCH_SIZE  # 1280
    crop_w = N_COLS * PATCH_SIZE  # 2048

    img  = img[:crop_h, :crop_w, :]   # (1280, 2048, 3)
    mask = mask[:crop_h, :crop_w, :]  # (1280, 2048, 4)

    img_4d  = tf.expand_dims(img,  0)  # (1, 1280, 2048, 3)
    mask_4d = tf.expand_dims(mask, 0)  # (1, 1280, 2048, 4)

    img_patches = tf.image.extract_patches(
        img_4d,
        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )  # (1, N_ROWS, N_COLS, PATCH_SIZE*PATCH_SIZE*3)
    img_patches = tf.reshape(img_patches, [N_PATCHES, PATCH_SIZE, PATCH_SIZE, 3])

    mask_patches = tf.image.extract_patches(
        mask_4d,
        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )  # (1, N_ROWS, N_COLS, PATCH_SIZE*PATCH_SIZE*4)
    mask_patches = tf.reshape(mask_patches, [N_PATCHES, PATCH_SIZE, PATCH_SIZE, NUM_CLASSES])

    return img_patches, mask_patches


# --- 4. Load one full-res image, return all its patches ---
def load_sample(image_path, rle_row):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32) / 255.0  # (1400, 2100, 3)

    def decode_masks(rle_row):
        mask = np.zeros((IMG_FULL_H, IMG_FULL_W, NUM_CLASSES), dtype=np.float32)
        for i, rle in enumerate(rle_row):
            rle_str = rle.numpy().decode('utf-8')
            if rle_str != '':
                mask[:, :, i] = rle2mask(rle_str, (IMG_FULL_H, IMG_FULL_W))
        return mask.astype(np.float32)

    masks = tf.py_function(decode_masks, [rle_row], tf.float32)
    masks.set_shape([IMG_FULL_H, IMG_FULL_W, NUM_CLASSES])

    img_patches, mask_patches = _extract_patches(img, masks)
    img_patches.set_shape([N_PATCHES, PATCH_SIZE, PATCH_SIZE, 3])
    mask_patches.set_shape([N_PATCHES, PATCH_SIZE, PATCH_SIZE, NUM_CLASSES])

    return img_patches, mask_patches


# --- 5. Build dataset ---
def build_dataset(df, image_dir, shuffle=True):
    image_paths = [f"{image_dir}/{fname}" for fname in df['ImageId']]
    rle_labels  = df[CLASSES].values

    path_tensor = tf.constant(image_paths, dtype=tf.string)
    rle_tensor  = tf.constant(rle_labels, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices((path_tensor, rle_tensor))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=500, seed=42)

    dataset = (
        dataset
        .map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .flat_map(lambda imgs, masks: tf.data.Dataset.from_tensor_slices((imgs, masks)))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset
