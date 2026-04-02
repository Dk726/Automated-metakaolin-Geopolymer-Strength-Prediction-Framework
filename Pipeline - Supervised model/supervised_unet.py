import os
import glob
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Configuration & Seed ---
seed_value = 19
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# --- 1. Data Preprocessing Functions ---

def process_images(path, l_path):
    """Processes 512x512 images"""
    images = []
    labels = []
    
    # Process Images
    for img_path in sorted(glob.glob(path)):
        img = cv2.imread(img_path, 0) # Grayscale
        if img is None: continue
        for i in [0, 256]:
            for j in [0, 256]:
                patch = img[i:i+256, j:j+256]
                images.append(patch)
                
    # Process Labels
    for lab_path in sorted(glob.glob(l_path)):
        lab = cv2.imread(lab_path, 0) # Grayscale
        if lab is None: continue
        for i in [0, 256]:
            for j in [0, 256]:
                patch = lab[i:i+256, j:j+256]
                labels.append(patch)
                
    return np.array(images), np.array(labels)

# Define paths (Adjust these to your Azure VM paths)
image_path = "path/to/images/*.png*" 
label_path = "path/to/labels/*.png*"

real_img, real_label = process_images(image_path, label_path)
print(f'Images: {real_img.shape}, Labels: {real_label.shape}')

# --- 2. Label Encoding ---

labelencoder = LabelEncoder()
n, h, w = real_label.shape
train_mask_reshaped = real_label.reshape(-1, 1)
train_mask_encoded = labelencoder.fit_transform(train_mask_reshaped.ravel())
train_mask = train_mask_encoded.reshape(n, h, w)

print(f"Unique classes: {np.unique(train_mask)}")

# Train/Test Split
X_train, X_test, Y_train, Y_test = train_test_split(
    real_img, train_mask, test_size=0.50, random_state=18, shuffle=True
)

# Expand dims for Grayscale channel (C=1)
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# --- 3. Custom Metrics ---

class MeanIoU_custom(tf.keras.metrics.Metric):
    def __init__(self, num_classes=5, name='mean_iou', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.iou_metric = tf.keras.metrics.MeanIoU(num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.argmax(y_pred, axis=-1, output_type=tf.int32)
        self.iou_metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.iou_metric.result()

    def reset_states(self):
        self.iou_metric.reset_state()

# --- 4. U-Net Model Definition ---

def conv_block(x, filters, dropout=0.0):
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    if dropout > 0:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def Unet_model(n_classes=5, W=256, H=256, C=1):
    inputs = tf.keras.layers.Input((W, H, C))
    
    # Encoder
    c1 = conv_block(inputs, 64, dropout=0.1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128, dropout=0.1)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256, dropout=0.2)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512, dropout=0.2)
    p4 = tf.keras.layers.MaxPooling2D((2, 2))(c4)

    c5 = conv_block(p4, 512, dropout=0.3) # Bottleneck

    # Decoder
    u6 = tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = conv_block(u6, 512, dropout=0.2)

    u7 = tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = conv_block(u7, 256, dropout=0.2)

    u8 = tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = conv_block(u8, 128, dropout=0.1)

    u9 = tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    c9 = conv_block(u9, 64, dropout=0.1)

    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1))(c9) # Logits output

    return tf.keras.Model(inputs=[inputs], outputs=[outputs])

# --- 5. Training ---

model = Unet_model(n_classes=5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[MeanIoU_custom(num_classes=5)]
)

cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_mean_iou", 
    patience=50, 
    verbose=1, 
    mode='max', 
    restore_best_weights=True
)

history = model.fit(
    X_train, Y_train,
    validation_data=(X_test, Y_test),
    callbacks=[cb],
    batch_size=32,
    epochs=3000,
    verbose=1,
    shuffle=False
)

# Save results
model.save('Supervised_Unet.keras')
pd.DataFrame(history.history).to_csv('supervised_unet_log.csv')

# --- 6. Visualization ---

def plot_unet_history(history_dict):
    df = pd.DataFrame(history_dict)
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (Log Scale)', fontsize=12)
    ax1.plot(df['loss'], label='Train Loss', color='tab:blue', lw=2)
    ax1.plot(df['val_loss'], label='Val Loss', color='tab:cyan', lw=2, ls='--')
    ax1.set_yscale('log')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('Mean IoU', fontsize=12)
    ax2.plot(df['mean_iou'], label='Train IoU', color='tab:green', lw=2)
    ax2.plot(df['val_mean_iou'], label='Val IoU', color='tab:red', lw=2, ls='--')
    ax2.set_ylim(0, 1.0)

    ax1.legend(loc='upper left', shadow=True)
    ax2.legend(loc='upper right', shadow=True)
    plt.title('U-Net Training Progress', fontsize=14)
    plt.show()

plot_unet_history(history.history)