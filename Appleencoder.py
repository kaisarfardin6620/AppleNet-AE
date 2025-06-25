import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras import layers, Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.svm import OneClassSVM

base_path = 'apple'
img_height, img_width = 224, 224
batch_size = 8 
epochs = 20

def load_images_from_folder(folder, img_height, img_width):
    valid_images = []
    
    if not os.path.exists(folder):
        raise ValueError(f"Folder {folder} does not exist")
    
    file_list = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(file_list)} image files in {folder}")
    
    for filename in file_list:
        img_path = os.path.join(folder, filename)
        try:
            with Image.open(img_path) as img:
                if img.mode == 'P':
                    img = img.convert('RGBA')
                img = img.convert('RGB')
                img = img.resize((img_width, img_height), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.float32)
                if img_array.shape != (img_height, img_width, 3):
                    print(f"Wrong shape {img_array.shape} for {filename}")
                    continue
                img_array = img_array / 255.0
                if np.any(np.isnan(img_array)) or np.any(np.isinf(img_array)):
                    print(f"Found NaN/Inf in {filename}, cleaning...")
                    img_array = np.nan_to_num(img_array, nan=0.0, posinf=1.0, neginf=0.0)
                if (img_array.size > 0 and 
                    np.all(img_array >= 0) and 
                    np.all(img_array <= 1) and
                    not np.any(np.isnan(img_array))):
                    valid_images.append(img_array)
                else:
                    print(f"Failed validation: {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    print(f"Successfully loaded {len(valid_images)} valid images from {folder}")
    
    if len(valid_images) == 0:
        raise ValueError(f"No valid images could be loaded from {folder}")
    
    try:
        images_array = np.stack(valid_images, axis=0)
        print(f"Final array shape: {images_array.shape}")
        print(f"Data range: [{np.min(images_array):.3f}, {np.max(images_array):.3f}]")
        if np.any(np.isnan(images_array)) or np.any(np.isinf(images_array)):
            print("WARNING: Still found invalid values, applying final cleanup...")
            images_array = np.nan_to_num(images_array, nan=0.0, posinf=1.0, neginf=0.0)
        return images_array
    except Exception as e:
        print(f"Error creating final array: {e}")
        raise

try:
    train_images = load_images_from_folder(os.path.join(base_path, 'train'), img_height, img_width)
    test_images = load_images_from_folder(os.path.join(base_path, 'test'), img_height, img_width)
except Exception as e:
    print(f"Error loading images: {e}")
    exit(1)

if len(train_images) < 2:
    raise ValueError("Not enough training images for train/validation split")

train_imgs, val_imgs = train_test_split(train_images, test_size=0.2, random_state=42)

print('Number of training samples:', len(train_imgs))
print('Number of testing samples:', len(test_images))
print('Number of validation samples:', len(val_imgs))

print("Creating TensorFlow datasets...")
print(f"Train data: shape={train_imgs.shape}, dtype={train_imgs.dtype}")
print(f"Val data: shape={val_imgs.shape}, dtype={val_imgs.dtype}")
print(f"Test data: shape={test_images.shape}, dtype={test_images.dtype}")

def validate_data_array(data, name):
    if data is None:
        raise ValueError(f"{name} is None")
    if len(data) == 0:
        raise ValueError(f"{name} is empty")
    if np.any(np.isnan(data)):
        raise ValueError(f"{name} contains NaN values")
    if np.any(np.isinf(data)):
        raise ValueError(f"{name} contains Inf values")
    print(f"{name} validation passed")

validate_data_array(train_imgs, "Training data")
validate_data_array(val_imgs, "Validation data")
validate_data_array(test_images, "Test data")

try:
    train_dataset = tf.data.Dataset.from_tensor_slices(train_imgs)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=False)
    print("Training dataset created successfully")
    val_dataset = tf.data.Dataset.from_tensor_slices(val_imgs)
    val_dataset = val_dataset.batch(batch_size, drop_remainder=False)
    print("Validation dataset created successfully")
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)
    test_dataset = test_dataset.batch(batch_size, drop_remainder=False)
    print("Test dataset created successfully")
except Exception as e:
    print(f"Error creating datasets: {e}")
    exit(1)

print("Testing dataset iteration...")
try:
    train_batch = next(iter(train_dataset))
    print(f"First training batch shape: {train_batch.shape}")
    print(f"First training batch range: [{tf.reduce_min(train_batch):.3f}, {tf.reduce_max(train_batch):.3f}]")
except Exception as e:
    print(f"Error iterating training dataset: {e}")
    exit(1)

def create_autoencoder(input_shape):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return autoencoder

print("Creating autoencoder model...")
autoencoder = create_autoencoder((img_height, img_width, 3))
autoencoder.summary()

print("Training autoencoder...")
print("Converting datasets to numpy arrays for training...")
try:
    X_train = train_imgs
    X_val = val_imgs
    print(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")
    history = autoencoder.fit(
        X_train, X_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, X_val),
        verbose=1,
        shuffle=True
    )
    print("Training completed successfully!")
except Exception as e:
    print(f"Numpy array training failed: {e}")
    print("Trying with batch_size=1...")
    try:
        history = autoencoder.fit(
            X_train, X_train,
            batch_size=1,
            epochs=epochs,
            validation_data=(X_val, X_val),
            verbose=1,
            shuffle=True
        )
        print("Training with batch_size=1 successful!")
    except Exception as e2:
        print(f"Even batch_size=1 failed: {e2}")
        print("Trying to recompile model with different optimizer...")
        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['mae']
        )
        try:
            history = autoencoder.fit(
                X_train, X_train,
                batch_size=4,
                epochs=epochs,
                validation_data=(X_val, X_val),
                verbose=1,
                shuffle=True
            )
            print("Training with recompiled model successful!")
        except Exception as e3:
            print(f"All training attempts failed. Final error: {e3}")
            exit(1)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training/Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Autoencoder Training/Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

print("Generating reconstructions...")
test_sample = test_images[:min(8, len(test_images))]
reconstructions = autoencoder.predict(test_sample, verbose=0)
n_display = min(5, len(test_sample))
plt.figure(figsize=(15, 6))
for i in range(n_display):
    plt.subplot(2, n_display, i + 1)
    plt.imshow(test_sample[i])
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, n_display, n_display + i + 1)
    plt.imshow(np.clip(reconstructions[i], 0, 1))
    plt.title('Reconstructed')
    plt.axis('off')
plt.tight_layout()
plt.show()

print("Training completed successfully!")

class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.mse = MeanSquaredError()

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(self.mse(data, reconstruction)) * (img_height * img_width)
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
            total_loss = reconstruction_loss + 0.1 * kl_loss  # Adjusted KL weight
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(self.mse(data, reconstruction)) * (img_height * img_width)
        kl_loss = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1))
        total_loss = reconstruction_loss + 0.1 * kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
        }

def create_vae_fixed(input_shape, latent_dim=32):
    encoder_inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', strides=2, padding='same')(encoder_inputs)
    x = layers.Conv2D(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    latent_inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense((img_height // 4) * (img_width // 4) * 64, activation='relu')(latent_inputs)
    x = layers.Reshape((img_height // 4, img_width // 4, 64))(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same')(x)
    decoder_outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    decoder = Model(latent_inputs, decoder_outputs, name='decoder')
    vae = VAE(encoder, decoder)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))
    return vae, encoder, decoder

print("\n--- Training Variational Autoencoder (VAE, fixed) ---")
vae, vae_encoder, vae_decoder = create_vae_fixed((img_height, img_width, 3), latent_dim=32)
vae.summary()
vae.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=1)

vae_recon = vae.predict(test_sample, verbose=0)
plt.figure(figsize=(15, 6))
for i in range(n_display):
    plt.subplot(2, n_display, i + 1)
    plt.imshow(test_sample[i])
    plt.title('Original')
    plt.axis('off')
    plt.subplot(2, n_display, n_display + i + 1)
    plt.imshow(np.clip(vae_recon[i], 0, 1))
    plt.title('VAE Recon')
    plt.axis('off')
plt.tight_layout()
plt.show()

encoder_model = Model(autoencoder.input, autoencoder.layers[4].output)
latent_features = encoder_model.predict(test_images, verbose=0)
latent_flat = latent_features.reshape(latent_features.shape[0], -1)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(latent_flat)
tsne = TSNE(n_components=2, random_state=42)
latent_2d = tsne.fit_transform(latent_flat)
plt.figure(figsize=(8, 6))
for i in range(n_clusters):
    idx = labels == i
    plt.scatter(latent_2d[idx, 0], latent_2d[idx, 1], label=f'Cluster {i}')
plt.title('t-SNE of Latent Space (Autoencoder)')
plt.legend()
plt.show()

def random_mask(images, mask_size=56):
    masked = images.copy()
    for img in masked:
        h, w = img.shape[:2]
        top = np.random.randint(0, h - mask_size)
        left = np.random.randint(0, w - mask_size)
        img[top:top+mask_size, left:left+mask_size, :] = 0.0
    return masked

masked_X_train = random_mask(X_train, mask_size=56)
masked_X_val = random_mask(X_val, mask_size=56)
inpaint_autoencoder = create_autoencoder((img_height, img_width, 3))
print("\n--- Training Autoencoder for Inpainting ---")
inpaint_autoencoder.fit(masked_X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(masked_X_val, X_val), verbose=1)
masked_test = random_mask(test_sample.copy(), mask_size=56)
inpaint_recon = inpaint_autoencoder.predict(masked_test, verbose=0)
plt.figure(figsize=(15, 9))
for i in range(n_display):
    plt.subplot(3, n_display, i + 1)
    plt.imshow(test_sample[i])
    plt.title('Original')
    plt.axis('off')
    plt.subplot(3, n_display, n_display + i + 1)
    plt.imshow(masked_test[i])
    plt.title('Masked')
    plt.axis('off')
    plt.subplot(3, n_display, 2 * n_display + i + 1)
    plt.imshow(np.clip(inpaint_recon[i], 0, 1))
    plt.title('Inpainted')
    plt.axis('off')
plt.tight_layout()
plt.show()

svm = OneClassSVM(gamma='auto').fit(latent_flat)
svm_scores = svm.score_samples(latent_flat)
plt.hist(svm_scores, bins=30)
plt.title('One-Class SVM Anomaly Scores (Test Set)')
plt.xlabel('Score (lower = more anomalous)')
plt.ylabel('Count')
plt.show()
n_anom = 5
anom_idx = np.argsort(svm_scores)[:n_anom]
plt.figure(figsize=(15, 3))
for i, idx in enumerate(anom_idx):
    plt.subplot(1, n_anom, i + 1)
    plt.imshow(test_images[idx])
    plt.title(f'Score: {svm_scores[idx]:.2f}')
    plt.axis('off')
plt.suptitle('Most Anomalous Test Images (SVM)')
plt.show()