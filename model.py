"""model.py
Cloud segmentation models for CubeSAT payload (target: Jetson Nano / TFLite).

All models:
  Input:  (batch, H, W, 3)  float32, normalized [0, 1]
  Output: (batch, H, W, 4)  float32, sigmoid — one binary mask per cloud class
  Loss:   BCE + Dice
  Metric: Dice coefficient

Models
------
  unet                     — U-Net from scratch (baseline)
  deeplabv3plus_pretrained — DeepLabV3+ with ImageNet MobileNetV2 backbone
  deeplabv3plus_scratch    — DeepLabV3+ with scratch MobileNetV2-style encoder
  segformer_pretrained     — SegFormer-B0 from HuggingFace (ADE20K pretrained)
  segformer_scratch        — Simplified MiT-B0 from scratch
  yolo_pretrained          — YOLOv8-inspired + ImageNet EfficientNetB0 backbone
  yolo_scratch             — YOLOv8-inspired CSP backbone from scratch
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 4
IMG_SIZE    = (256, 384)  # (H, W)


# ============================================================
# Loss & Metrics
# ============================================================

def dice_loss(y_true, y_pred, smooth=1e-6):
    """Soft Dice loss averaged over batch and class channels."""
    y_true = tf.cast(y_true, tf.float32)
    # Sum over H, W → per-(sample, class) overlap
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union        = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    return 1.0 - tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


def bce_dice_loss(y_true, y_pred):
    """BCE + Dice: stable gradients + handles sparse cloud masks."""
    bce = tf.reduce_mean(keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)


def dice_coefficient(y_true, y_pred, threshold=0.5, smooth=1e-6):
    """Hard-thresholded Dice coefficient (use as metric, not as loss)."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union        = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    return tf.reduce_mean((2.0 * intersection + smooth) / (union + smooth))


# ============================================================
# Shared Building Blocks
# ============================================================

def _conv_bn_relu(x, filters, kernel_size=3, stride=1, dilation_rate=1):
    x = layers.Conv2D(
        filters, kernel_size, strides=stride, padding='same',
        dilation_rate=dilation_rate, use_bias=False
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


# ============================================================
# 1. U-Net  (scratch baseline)
# ============================================================

def _unet_block(x, filters):
    x = _conv_bn_relu(x, filters)
    x = _conv_bn_relu(x, filters)
    return x


def build_unet(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    Standard U-Net — 4 encoder levels, bottleneck, 4 decoder levels.
    Skip connections preserve spatial detail lost during downsampling.
    ~7.7M params.  Scratch baseline.
    """
    inp = keras.Input(shape=(*img_size, 3), name='input')

    # Encoder
    e1 = _unet_block(inp, 32);  p1 = layers.MaxPooling2D()(e1)
    e2 = _unet_block(p1,  64);  p2 = layers.MaxPooling2D()(e2)
    e3 = _unet_block(p2, 128);  p3 = layers.MaxPooling2D()(e3)
    e4 = _unet_block(p3, 256);  p4 = layers.MaxPooling2D()(e4)

    # Bottleneck
    b = _unet_block(p4, 512)

    # Decoder — upsample, concat skip, conv block
    def _up(x, skip, f):
        x = layers.UpSampling2D(interpolation='bilinear')(x)
        x = layers.Concatenate()([x, skip])
        return _unet_block(x, f)

    d = _up(b,  e4, 256)
    d = _up(d,  e3, 128)
    d = _up(d,  e2,  64)
    d = _up(d,  e1,  32)

    out = layers.Conv2D(num_classes, 1, activation='sigmoid', name='output')(d)
    return keras.Model(inp, out, name='unet_scratch')


# ============================================================
# 2. DeepLabV3+  (pretrained + scratch)
# ============================================================

def _aspp(x, filters=256):
    """
    Atrous Spatial Pyramid Pooling.
    Parallel dilated convolutions at multiple rates capture multi-scale context
    without reducing spatial resolution.
    """
    h, w = tf.shape(x)[1], tf.shape(x)[2]

    b0 = _conv_bn_relu(x, filters, kernel_size=1)       # 1×1 context
    b1 = _conv_bn_relu(x, filters, dilation_rate=6)     # rate 6
    b2 = _conv_bn_relu(x, filters, dilation_rate=12)    # rate 12
    b3 = _conv_bn_relu(x, filters, dilation_rate=18)    # rate 18

    # Global average pooling branch — image-level context
    b4 = layers.GlobalAveragePooling2D()(x)
    b4 = layers.Reshape((1, 1, -1))(b4)
    b4 = layers.Conv2D(filters, 1, use_bias=False)(b4)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.ReLU()(b4)
    b4 = tf.image.resize(b4, [h, w], method='bilinear')

    x = layers.Concatenate()([b0, b1, b2, b3, b4])
    x = _conv_bn_relu(x, filters, kernel_size=1)
    return x


def build_deeplabv3plus_pretrained(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    DeepLabV3+ with ImageNet-pretrained MobileNetV2 backbone.
      Low-level features:  stride-4  → block_1_project_BN  (H/4,  24ch)
      High-level features: stride-32 → out_relu             (H/32, 1280ch)
    ASPP fuses multi-scale context, decoder refines boundaries.
    ~4.5M params.

    NOTE: if a ValueError occurs on layer names, run:
          tf.keras.applications.MobileNetV2(..., include_top=False).summary()
    """
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*img_size, 3), include_top=False, weights='imagenet'
    )
    backbone.trainable = True

    low_level  = backbone.get_layer('block_1_project_BN').output  # H/4,  W/4,  24ch
    high_level = backbone.get_layer('out_relu').output            # H/32, W/32, 1280ch
    feat_model = keras.Model(backbone.input, [low_level, high_level])

    inp    = keras.Input(shape=(*img_size, 3), name='input')
    ll, hl = feat_model(inp)

    # ASPP on high-level features then upsample 8× to reach stride-4 resolution
    x  = _aspp(hl, filters=256)
    x  = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)

    # Project low-level to 48ch (standard DeepLabV3+ ratio)
    ll = _conv_bn_relu(ll, 48, kernel_size=1)
    x  = layers.Concatenate()([x, ll])
    x  = _conv_bn_relu(x, 256)
    x  = _conv_bn_relu(x, 256)

    # Upsample 4× to full resolution
    x   = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    out = layers.Conv2D(num_classes, 1, activation='sigmoid', name='output')(x)
    return keras.Model(inp, out, name='deeplabv3plus_pretrained')


def _inverted_residual(x, filters, expand_ratio=6, stride=1):
    """MobileNetV2-style inverted residual block (depthwise separable convs)."""
    in_ch    = x.shape[-1]
    residual = x

    if expand_ratio != 1:
        x = _conv_bn_relu(x, in_ch * expand_ratio, kernel_size=1)

    x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride == 1 and in_ch == filters:
        x = layers.Add()([x, residual])
    return x


def build_deeplabv3plus_scratch(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    DeepLabV3+ with a from-scratch MobileNetV2-style encoder.
    Same ASPP + decoder as the pretrained version.
    ~2.1M params.
    """
    inp = keras.Input(shape=(*img_size, 3), name='input')
    x   = _conv_bn_relu(inp, 16, stride=2)  # stem → H/2

    # (out_filters, expand_ratio, stride, repeats)
    spec = [
        (16,  1, 1, 1),  # H/2
        (24,  6, 2, 2),  # H/4  ← low-level features
        (32,  6, 2, 3),  # H/8
        (64,  6, 2, 4),  # H/16
        (96,  6, 1, 3),  # H/16
        (160, 6, 2, 3),  # H/32
        (320, 6, 1, 1),  # H/32
    ]

    low_level = None
    for i, (f, t, s, n) in enumerate(spec):
        for j in range(n):
            x = _inverted_residual(x, f, expand_ratio=t, stride=s if j == 0 else 1)
        if i == 1:  # just finished the stride-4 group
            low_level = x

    x  = _aspp(x, filters=256)
    x  = layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(x)
    ll = _conv_bn_relu(low_level, 48, kernel_size=1)
    x  = layers.Concatenate()([x, ll])
    x  = _conv_bn_relu(x, 256)
    x  = _conv_bn_relu(x, 256)
    x  = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    out = layers.Conv2D(num_classes, 1, activation='sigmoid', name='output')(x)
    return keras.Model(inp, out, name='deeplabv3plus_scratch')


# ============================================================
# 3. SegFormer-B0  (pretrained + scratch)
# ============================================================

class _SegFormerWrapper(layers.Layer):
    """Wraps a HuggingFace TFSegformerForSemanticSegmentation as a Keras layer."""
    def __init__(self, hf_model, img_size, **kwargs):
        super().__init__(**kwargs)
        self.hf_model = hf_model
        self.img_size = img_size

    def call(self, x, training=False):
        x      = tf.transpose(x, [0, 3, 1, 2])               # (B,H,W,C) → (B,C,H,W)
        logits = self.hf_model(x, training=training).logits  # (B, num_classes, H/4, W/4)
        logits = tf.transpose(logits, [0, 2, 3, 1])          # → (B, H/4, W/4, num_classes)
        return logits


def build_segformer_pretrained(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    SegFormer-B0 from HuggingFace — ADE20K pretrained, head replaced for num_classes.
    ~3.7M params.

    Requires: pip install transformers

    NOTE: TFLite export of HuggingFace models may require an ONNX intermediate
          step. Test export on a small input before deploying.
    """
    from transformers import TFSegformerForSemanticSegmentation, SegformerConfig

    config   = SegformerConfig.from_pretrained(
        'nvidia/segformer-b0-finetuned-ade-512-512',
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    hf_model = TFSegformerForSemanticSegmentation.from_pretrained(
        'nvidia/segformer-b0-finetuned-ade-512-512',
        config=config,
        ignore_mismatched_sizes=True,
    )

    inp    = keras.Input(shape=(*img_size, 3), name='input')
    logits = _SegFormerWrapper(hf_model, img_size)(inp)        # (B, H/4, W/4, C)
    logits = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(logits)
    out    = layers.Activation('sigmoid', name='output')(logits)
    return keras.Model(inp, out, name='segformer_pretrained')


# --- SegFormer scratch sub-layers ---

class _EfficientSelfAttention(layers.Layer):
    """
    Mix Transformer efficient self-attention.
    Spatial reduction (sr_ratio) shrinks the key/value sequence length,
    keeping attention cost manageable at high-resolution stages.
    """
    def __init__(self, dim, num_heads, sr_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.sr_ratio  = sr_ratio
        self.dim       = dim

        self.q    = layers.Dense(dim)
        self.kv   = layers.Dense(dim * 2)
        self.proj = layers.Dense(dim)

        if sr_ratio > 1:
            self.sr   = layers.Conv2D(dim, sr_ratio, strides=sr_ratio, use_bias=False)
            self.norm = layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, h, w, training=False):
        B  = tf.shape(x)[0]
        N  = h * w
        nh = self.num_heads
        hd = self.head_dim

        q = self.q(x)
        q = tf.reshape(q, [B, N, nh, hd])
        q = tf.transpose(q, [0, 2, 1, 3])  # (B, nh, N, hd)

        # Reduce spatial resolution of keys/values
        if self.sr_ratio > 1:
            x_ = tf.reshape(x, [B, h, w, self.dim])
            x_ = self.sr(x_)
            x_ = tf.reshape(x_, [B, -1, self.dim])
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_)
        kv = tf.reshape(kv, [B, -1, 2, nh, hd])
        kv = tf.transpose(kv, [2, 0, 3, 1, 4])
        k, v = kv[0], kv[1]  # each (B, nh, N', hd)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale
        attn = tf.nn.softmax(attn, axis=-1)
        x    = tf.matmul(attn, v)          # (B, nh, N, hd)
        x    = tf.transpose(x, [0, 2, 1, 3])
        x    = tf.reshape(x, [B, N, nh * hd])
        return self.proj(x)


class _MixFFN(layers.Layer):
    """
    Mix-FFN: linear expand → depthwise conv (local context) → linear project.
    GELU activation (standard in transformers).
    """
    def __init__(self, dim, mlp_ratio=4, **kwargs):
        super().__init__(**kwargs)
        self.hidden = int(dim * mlp_ratio)
        self.fc1    = layers.Dense(self.hidden)
        self.dw     = layers.DepthwiseConv2D(3, padding='same')
        self.fc2    = layers.Dense(dim)
        self.act    = layers.Activation('gelu')

    def call(self, x, h, w, training=False):
        B = tf.shape(x)[0]
        x = self.fc1(x)
        x = tf.reshape(x, [B, h, w, self.hidden])
        x = self.dw(x)
        x = tf.reshape(x, [B, h * w, self.hidden])
        x = self.act(x)
        return self.fc2(x)


class _MiTBlock(layers.Layer):
    """One Mix Transformer encoder block: efficient attention + Mix-FFN."""
    def __init__(self, dim, num_heads, mlp_ratio=4, sr_ratio=1, **kwargs):
        super().__init__(**kwargs)
        self.norm1 = layers.LayerNormalization(epsilon=1e-5)
        self.norm2 = layers.LayerNormalization(epsilon=1e-5)
        self.attn  = _EfficientSelfAttention(dim, num_heads, sr_ratio)
        self.ffn   = _MixFFN(dim, mlp_ratio)

    def call(self, x, h, w, training=False):
        x = x + self.attn(self.norm1(x), h, w, training=training)
        x = x + self.ffn(self.norm2(x),  h, w, training=training)
        return x


class SegFormerScratch(keras.Model):
    """
    Simplified MiT-B0 encoder + all-MLP decoder, built from scratch.
    4 hierarchical stages with overlapping patch embeddings.
    ~1.8M params.
    """
    def __init__(self, img_size=IMG_SIZE, num_classes=NUM_CLASSES, **kwargs):
        super().__init__(name='segformer_scratch', **kwargs)
        H, W = img_size

        # Stage configs: (embed_dim, num_heads, sr_ratio, depth, patch_size, stride)
        cfgs = [
            (32,  1, 8, 2, 7, 4),  # Stage 1: H/4,  W/4
            (64,  2, 4, 2, 3, 2),  # Stage 2: H/8,  W/8
            (160, 5, 2, 2, 3, 2),  # Stage 3: H/16, W/16
            (256, 8, 1, 2, 3, 2),  # Stage 4: H/32, W/32
        ]
        self.stage_sizes = [
            (H // 4,  W // 4),
            (H // 8,  W // 8),
            (H // 16, W // 16),
            (H // 32, W // 32),
        ]

        decoder_dim = 256
        self.patch_embeds  = []
        self.patch_norms   = []
        self.stage_blocks  = []
        self.stage_norms   = []
        self.decoder_proj  = []

        for dim, heads, sr, depth, ps, st in cfgs:
            self.patch_embeds.append(
                layers.Conv2D(dim, ps, strides=st, padding='same', use_bias=False)
            )
            self.patch_norms.append(layers.LayerNormalization(epsilon=1e-5))
            self.stage_blocks.append(
                [_MiTBlock(dim, heads, sr_ratio=sr) for _ in range(depth)]
            )
            self.stage_norms.append(layers.LayerNormalization(epsilon=1e-5))
            self.decoder_proj.append(layers.Conv2D(decoder_dim, 1, use_bias=False))

        self.decoder_fuse = layers.Conv2D(decoder_dim, 1, use_bias=False)
        self.decoder_bn   = layers.BatchNormalization()
        self.decoder_relu = layers.ReLU()
        self.dropout      = layers.Dropout(0.1)
        self.head         = layers.Conv2D(num_classes, 1, activation='sigmoid')
        self.img_size     = img_size

    def call(self, x, training=False):
        features = []

        for i in range(4):
            h, w = self.stage_sizes[i]
            B    = tf.shape(x)[0]
            dim  = self.patch_embeds[i].filters

            # Overlapping patch embedding
            x = self.patch_embeds[i](x)               # (B, h, w, dim)
            x = tf.reshape(x, [B, h * w, dim])
            x = self.patch_norms[i](x)

            # Transformer blocks
            for block in self.stage_blocks[i]:
                x = block(x, h, w, training=training)
            x = self.stage_norms[i](x)

            # Reshape back to spatial
            x = tf.reshape(x, [B, h, w, dim])
            features.append(x)

        # All-MLP decoder: project all stages → decoder_dim, upsample to H/4, fuse
        target_h = self.img_size[0] // 4
        target_w = self.img_size[1] // 4

        fused = [
            tf.image.resize(self.decoder_proj[i](f), [target_h, target_w], method='bilinear')
            for i, f in enumerate(features)
        ]
        x = tf.concat(fused, axis=-1)
        x = self.decoder_fuse(x)
        x = self.decoder_bn(x, training=training)
        x = self.decoder_relu(x)
        x = self.dropout(x, training=training)
        x = tf.image.resize(x, self.img_size, method='bilinear')
        return self.head(x)


def build_segformer_scratch(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    return SegFormerScratch(img_size=img_size, num_classes=num_classes)


# ============================================================
# 4. YOLOv8-Inspired Semantic Segmentation  (pretrained + scratch)
# ============================================================

def _csp_block(x, filters, num_bottlenecks=1):
    """
    Cross Stage Partial block.
    Splits channels, runs bottlenecks on one half, concatenates both.
    Reduces computation while preserving gradient flow.
    """
    half = filters // 2
    x1   = _conv_bn_relu(x, half, kernel_size=1)   # skip path
    x2   = _conv_bn_relu(x, half, kernel_size=1)   # CSP path
    for _ in range(num_bottlenecks):
        res = x2
        x2  = _conv_bn_relu(x2, half)
        x2  = _conv_bn_relu(x2, half)
        x2  = layers.Add()([x2, res])
    x = layers.Concatenate()([x1, x2])
    x = _conv_bn_relu(x, filters, kernel_size=1)
    return x


def _fpn_neck(c3, c4, c5, filters=128):
    """
    PANet FPN neck (YOLOv8-style).
    Top-down path: merges high-level context downward (c5 → c4 → c3).
    Bottom-up path: refines with local features upward (c3 → c4 → c5).
      c3: H/8   c4: H/16   c5: H/32
    """
    # Top-down
    p5 = _conv_bn_relu(c5, filters, kernel_size=1)
    p4 = layers.UpSampling2D(interpolation='bilinear')(p5)
    p4 = layers.Concatenate()([p4, _conv_bn_relu(c4, filters, kernel_size=1)])
    p4 = _csp_block(p4, filters)

    p3 = layers.UpSampling2D(interpolation='bilinear')(p4)
    p3 = layers.Concatenate()([p3, _conv_bn_relu(c3, filters, kernel_size=1)])
    p3 = _csp_block(p3, filters)

    # Bottom-up
    n3 = p3
    n4 = layers.Concatenate()([_conv_bn_relu(n3, filters, stride=2), p4])
    n4 = _csp_block(n4, filters)
    n5 = layers.Concatenate()([_conv_bn_relu(n4, filters, stride=2), p5])
    n5 = _csp_block(n5, filters)

    return n3, n4, n5


def _semantic_head(n3, n4, n5, img_size, num_classes):
    """Upsample all FPN outputs to H/4, fuse, then upsample to full resolution."""
    target_h, target_w = img_size[0] // 4, img_size[1] // 4
    n3  = tf.image.resize(n3, [target_h, target_w], method='bilinear')
    n4  = tf.image.resize(n4, [target_h, target_w], method='bilinear')
    n5  = tf.image.resize(n5, [target_h, target_w], method='bilinear')
    x   = layers.Concatenate()([n3, n4, n5])
    x   = _conv_bn_relu(x, 128)
    x   = layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    return layers.Conv2D(num_classes, 1, activation='sigmoid', name='output')(x)


def build_yolo_semantic_pretrained(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    YOLOv8-inspired semantic segmentation with pretrained EfficientNetB0 backbone.
    EfficientNetB0 feature extraction:
      C3 → block4a_expand_activation  (H/8,  W/8)
      C4 → block6a_expand_activation  (H/16, W/16)
      C5 → top_activation             (H/32, W/32)
    PANet FPN neck + semantic head.
    ~5.3M params.

    NOTE: if a ValueError occurs on layer names, run:
          tf.keras.applications.EfficientNetB0(..., include_top=False).summary()
    """
    backbone = tf.keras.applications.EfficientNetB0(
        input_shape=(*img_size, 3), include_top=False, weights='imagenet'
    )
    backbone.trainable = True

    c3 = backbone.get_layer('block4a_expand_activation').output  # H/8
    c4 = backbone.get_layer('block6a_expand_activation').output  # H/16
    c5 = backbone.get_layer('top_activation').output             # H/32
    feat_model = keras.Model(backbone.input, [c3, c4, c5])

    inp        = keras.Input(shape=(*img_size, 3), name='input')
    c3, c4, c5 = feat_model(inp)
    n3, n4, n5  = _fpn_neck(c3, c4, c5, filters=128)
    out         = _semantic_head(n3, n4, n5, img_size, num_classes)
    return keras.Model(inp, out, name='yolo_semantic_pretrained')


def build_yolo_semantic_scratch(num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    YOLOv8-inspired semantic segmentation, fully from scratch.
    CSP backbone → PANet FPN neck → semantic head.
    ~3.1M params.
    """
    inp = keras.Input(shape=(*img_size, 3), name='input')

    # Stem
    x = _conv_bn_relu(inp, 32, stride=2)   # H/2
    x = _conv_bn_relu(x,   64, stride=2)   # H/4

    # Backbone
    c3 = _csp_block(_conv_bn_relu(x,  128, stride=2), 128, num_bottlenecks=2)  # H/8
    c4 = _csp_block(_conv_bn_relu(c3, 256, stride=2), 256, num_bottlenecks=4)  # H/16
    c5 = _csp_block(_conv_bn_relu(c4, 512, stride=2), 512, num_bottlenecks=2)  # H/32

    n3, n4, n5 = _fpn_neck(c3, c4, c5, filters=128)
    out        = _semantic_head(n3, n4, n5, img_size, num_classes)
    return keras.Model(inp, out, name='yolo_semantic_scratch')


# ============================================================
# Model Registry
# ============================================================

def get_model(name, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
    """
    Instantiate a model by name.

    Available models:
      'unet'                     — U-Net from scratch (baseline)
      'deeplabv3plus_pretrained' — DeepLabV3+ + ImageNet MobileNetV2
      'deeplabv3plus_scratch'    — DeepLabV3+ from scratch
      'segformer_pretrained'     — SegFormer-B0 from HuggingFace
      'segformer_scratch'        — Simplified MiT-B0 from scratch
      'yolo_pretrained'          — YOLOv8-inspired + ImageNet EfficientNetB0
      'yolo_scratch'             — YOLOv8-inspired from scratch

    Example:
        model = get_model('unet')
        model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coefficient])
    """
    registry = {
        'unet':                     build_unet,
        'deeplabv3plus_pretrained': build_deeplabv3plus_pretrained,
        'deeplabv3plus_scratch':    build_deeplabv3plus_scratch,
        'segformer_pretrained':     build_segformer_pretrained,
        'segformer_scratch':        build_segformer_scratch,
        'yolo_pretrained':          build_yolo_semantic_pretrained,
        'yolo_scratch':             build_yolo_semantic_scratch,
    }
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(registry)}")
    return registry[name](num_classes=num_classes, img_size=img_size)
