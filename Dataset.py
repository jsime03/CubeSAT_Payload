import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

IMG_SIZE = (256, 384)
BATCH_SIZE = 8
NUM_CLASSES = 4
CLASSES = ['Fish', 'Flower', 'Gravel', 'Sugar']

# --- 1. Parse CSV (exactly as in torch code) ---
def parse_csv(csv_path):
    train_df = pd.read_csv(csv_path)

    train_df[['ImageId', 'Label']] = train_df['Image_Label'].str.split('_', expand=True)
    train_df = train_df.drop(columns=['Image_Label'])
    train_df['EncodedPixels'] = train_df['EncodedPixels'].fillna('')
    train_df = train_df.pivot(index='ImageId', columns='Label', values='EncodedPixels').reset_index()
    train_df.fillna('', inplace=True)

    # --- 2. KFold split ---
    train_df['fold'] = -1
    kf = KFold(n_splits=9, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        train_df.loc[val_idx, 'fold'] = fold

    train_data = train_df[train_df['fold'] != 0].reset_index(drop=True)
    val_data   = train_df[train_df['fold'] == 0].reset_index(drop=True)
    return train_data, val_data


# --- 3. RLE decode (same logic as rle2mask) ---
def rle2mask(rle, shape=(1400, 2100)):
    if rle == '':
        return np.zeros(shape, dtype=np.float32)
    s = rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1  # zero-based indexing
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # column-major like torch version

# --- 4. Load one sample ---
def load_sample(image_path, rle_row):
    # Load and normalize image (replaces cv2.imread + transpose + /255)
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0

    # Decode masks (replaces the for loop over labels)
    def decode_masks(rle_row):
        mask = np.zeros((1400, 2100, 4), dtype=np.float32)
        for i, rle in enumerate(rle_row):
            rle_str = rle.numpy().decode('utf-8')
            if rle_str != '':
                mask[:, :, i] = rle2mask(rle_str, (1400, 2100))
        # Resize (replaces cv2.resize)
        mask = tf.image.resize(mask, IMG_SIZE).numpy()
        return mask.astype(np.float32)

    masks = tf.py_function(decode_masks, [rle_row], tf.float32)
    masks.set_shape([IMG_SIZE[0], IMG_SIZE[1], NUM_CLASSES])

    # Note: no transpose needed — TF uses channel-last (H, W, C) unlike PyTorch

    return img, masks

# --- 5. Build dataset from a dataframe ---
def build_dataset(df, image_dir, shuffle=True):
    image_paths = [f"{image_dir}/{fname}" for fname in df['ImageId']]
    rle_labels  = df[CLASSES].values  # (N, 4)

    path_tensor = tf.constant(image_paths, dtype=tf.string)
    rle_tensor  = tf.constant(rle_labels, dtype=tf.string)

    dataset = tf.data.Dataset.from_tensor_slices((path_tensor, rle_tensor))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=500, seed=42)

    dataset = (
        dataset
        .map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return dataset

# --- 6. Usage ---

# --- 5. Usage ---
# train_ds = build_dataset('train.csv', 'train_images/', shuffle=True)

