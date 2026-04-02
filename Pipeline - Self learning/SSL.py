import glob
import random
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from numpy.random import seed, randint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import mixed_precision

# --- INITIAL CONFIGURATION ---
seed(19)
tf.random.set_seed(19)

# Mixed Precision for performance
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# --- DATA PROCESSING ---
def process_images(path, l_path): #Currently set to process 512x512 images
    images = []
    labels = []
    for img_path in sorted(glob.glob(path)):
        img = cv2.imread(img_path, 0)
        for i in (0, 256):
            for j in (0, 256):
                patch = img[i:i+256, j:j+256]
                images.append(patch)
    for lab_path in sorted(glob.glob(l_path)):
        lab = cv2.imread(lab_path, 0)
        for i in (0, 256):
            for j in (0, 256):
                patch = lab[i:i+256, j:j+256]
                labels.append(patch)
    return images, labels

# Define paths and process
image_path = "path/to/images/*.png*" 
label_path = "path/to/labels/*.png*"

img, label = process_images(image_path, label_path)
real_img = np.array(img)
real_label = np.array(label)
print('Shape of Image array:', np.shape(real_img), '\nShape of Label array:', np.shape(real_label))

# --- LABEL ENCODING ---
labelencoder = LabelEncoder()
n, h, w = real_label.shape
train_mask_reshaped = real_label.reshape(-1,1)
train_mask_encoded = labelencoder.fit_transform(train_mask_reshaped)
train_mask = train_mask_encoded.reshape(n, h, w)
print(f"Unique classes: {np.unique(train_mask)}")

# --- TRAIN TEST SPLIT ---
X_train, X_test, Y_train, Y_test = train_test_split(
    real_img, train_mask, test_size=0.50, random_state=18, shuffle=True
)

# --- GAN / SYNTHETIC DATA LOADING ---
fake_imgs = []
fake_path = 'path/to/synthetic images/*.*' #Currently processing 512x512 synthetic images
for imgs in glob.glob(fake_path):
    img = cv2.imread(imgs, 0)
    for i in range(0, 512, 256):
        for j in range(0, 512, 256):
            patch = img[i:i+256, j:j+256]
            fake_imgs.append(patch)
            
fake_imgs = np.array(fake_imgs)
print('Shape of synthetic image array:', np.shape(fake_imgs))

# --- CUSTOM METRICS ---
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

def load_real_samples(real_img, real_label, batch):
    select = randint(0, np.shape(real_img)[0]-1, batch)
    X_real, Y_real = real_img[select], real_label[select]
    return X_real, Y_real

# --- VISUALIZATION SETUP ---
custom_rgb = [
    (128, 128, 128), (17, 37, 100), (208, 100, 88), 
    (195, 195, 201), (225, 231, 109), (170, 0, 255)
]
custom_cmap = ListedColormap([np.array(c)/255.0 for c in custom_rgb])

def plot_ssl_history(history, n, folder, ssl=True):
    rounds = history['round'] if ssl else history['epochs']
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs/Rounds')
    ax1.set_ylabel('IoU Score')
    ax1.plot(rounds, history['train_iou'], label='Train IoU', marker='o', color='blue')
    ax1.plot(rounds, history['test_iou'], label='Test IoU', marker='o', color='green')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    if ssl:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Confident Pixel %', color='red')
        ax2.plot(rounds, history['conf_pixel_pct'], label='Conf Pixel %', linestyle='--', color='red', alpha=0.6)
        plt.title('Self-Learning Progress: IoU and Pseudo-Label Confidence')
        plt.savefig(f'{folder}/SSL progress_round_{n}.png')
    plt.show()

def plot_pseudo_label_dist(history, n, folder):
    rounds = history['round']
    c_keys = [f'conf_class_{i}' for i in range(5)]
    data = [np.array(history[k]) for k in c_keys]
    totals = sum(data)
    percentages = [(d / totals) * 100 for d in data]
    
    plt.figure(figsize=(10, 6))
    bottoms = np.zeros(len(rounds))
    colors = ['#112564', '#e1e76d', '#c3c3c9', '#d06458', '#000000']
    labels = ['Porosity', 'Gel', 'Aggregate', 'Unreacted', 'Impurities']
    
    for p, c, l in zip(percentages, colors, labels):
        plt.bar(rounds, p, bottom=bottoms, label=l, color=c)
        bottoms += p

    plt.xlabel('Round Number')
    plt.ylabel('Percentage (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{folder}/psudo dist_round_{n}.png')
    plt.show()

# --- SSL FUNCTIONS ---
@tf.function
def masked_pseudo_loss(y_true, y_pred_logits, weight_mask):
    pixel_loss = base_loss_fn(y_true, y_pred_logits)
    masked_loss = pixel_loss * weight_mask
    num_confident_pixels = tf.reduce_sum(weight_mask)
    return tf.where(num_confident_pixels > 0, tf.reduce_sum(masked_loss) / num_confident_pixels, 0.0)

@tf.function
def train_step_pseudo(model, x, y, w, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = masked_pseudo_loss(y, logits, w)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

@tf.function
def train_step_labeled(model, x, y, optimizer):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        loss = masked_pseudo_loss(y, logits, tf.ones_like(y, dtype=tf.float32)) * 2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return logits, loss

@tf.function
def predict_step(model, batch, T):
    logits = model(batch, training=False)
    return tf.nn.softmax(logits / T, axis=-1)

def get_probs(model, data, T, batch_size=8):
    predictions = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        p = predict_step(model, batch, T)
        predictions.append(p.numpy())
    return np.concatenate(predictions, axis=0)

# --- TRAINING LOOP ---
def self_learning_training(model, X_L, Y_L, x_test, y_test, X_U, optimizer, folder='output', 
                           num_rounds=5, start_round=0, retrain_epochs=3, 
                           confidence_threshold=0.9, batch_size=32, patience=3, T=1):
    X_unlabeled_images = X_U
    history = {k: [] for k in ['round', 'conf_pixel_pct', 'train_iou', 'test_iou']}
    history.update({f'conf_class_{i}': [] for i in range(5)})
    history.update({f'test_iou_{i}': [] for i in range(5)})
    
    best_test_iou, wait = 0.0, 0
    best_weights = model.get_weights() 
    train_iou_metric = MeanIoU_custom(num_classes=5)
    test_iou_metric = MeanIoU_custom(num_classes=5)

    for round_num in range(start_round, num_rounds):
        if (round_num + 1) >= 15: confidence_threshold = 0.99
        
        # Inference & Pseudo-label Generation
        probs = get_probs(model, X_unlabeled_images, T)
        max_probs = np.max(probs, axis=-1)
        preds = np.argmax(probs, axis=-1)
        confident_mask = (max_probs >= confidence_threshold).astype(np.float32)
        weight_mask = confident_mask * max_probs 
        
        confident_indices = np.where(np.sum(confident_mask, axis=(1, 2)) > 0)[0]
        if confident_indices.size == 0: break

        X_P, Y_P, W_P = X_unlabeled_images[confident_indices], preds[confident_indices], weight_mask[confident_indices]
        unique, counts = np.unique(Y_P, return_counts=True)
        dist = dict(zip(unique, counts))

        # Training Phase
        model.trainable = True
        batches_per_epoch = int(X_P.shape[0] / batch_size) 
        for e in range(retrain_epochs):
            train_iou_metric.reset_states()
            for j in range(batches_per_epoch):
                sel_p = randint(0, X_P.shape[0] - 1, batch_size)
                train_step_pseudo(model, X_P[sel_p], Y_P[sel_p], W_P[sel_p], optimizer)
                xb_l, yb_l = load_real_samples(X_L, Y_L, batch_size)
                logits_l, _ = train_step_labeled(model, xb_l, yb_l, optimizer)
                train_iou_metric.update_state(yb_l, tf.nn.softmax(logits_l, axis=-1))

        # Evaluation
        test_iou_metric.reset_states()
        test_pred = model.predict(x_test, batch_size=16, verbose=0)
        test_iou_metric.update_state(y_test, tf.nn.softmax(test_pred, axis=-1))
        
        # Logging
        curr_test_iou = test_iou_metric.result().numpy()
        history['round'].append(round_num + 1)
        history['test_iou'].append(curr_test_iou)
        # (Remaining history updates and checkpoint logic...)

        if curr_test_iou > best_test_iou:
            best_test_iou = curr_test_iou
            wait = 0; best_weights = model.get_weights() 
        else:
            wait += 1
            if wait >= patience: break

    model.set_weights(best_weights)
    return history

# --- EXECUTION ---
model = keras.saving.load_model('supervised_model.keras', custom_objects={'MeanIoU_custom': MeanIoU_custom}) #Load trained supervised model here
optimizer = Adam(learning_rate=0.0001)
base_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

ssl_history = self_learning_training(
    model, X_train, Y_train, X_test, Y_test, fake_imgs, 
    optimizer, folder='output', num_rounds=1000, #Change folder name as per requirement
    retrain_epochs=3, confidence_threshold=0.99, batch_size=16, patience=50, T=2 #T is temperature scaling
)