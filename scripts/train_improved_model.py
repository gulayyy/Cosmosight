import os, sys, json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMAGE_SIZE, TRAIN_DIR, VAL_DIR, CLASS_NAMES_PATH, HISTORY_PATH

# =================  IMPROVED SETTINGS  =================
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Model parameters
BATCH_SIZE = 16  # Smaller batch size for more stable training
BASE_LR = 1e-4   # More conservative learning rate
EPOCHS_STAGE1 = 8   # Fewer epochs with frozen base
EPOCHS_STAGE2 = 22  # More epochs for fine-tuning
LABEL_SMOOTH = 0.1  # More aggressive label smoothing

# New model save paths
MODEL_SAVE_PATH = "models/astro_model_improved.h5"
KERAS_SAVE_PATH = "models/astro_model_improved.keras"
IMPROVED_HISTORY_PATH = "models/improved_training_history.npy"
IMPROVED_LOG_PATH = "models/improved_training_log.csv"

# GPU optimizasyonu
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass
print("🔧 Using GPU:" if gpus else "🔧 Using CPU:", len(gpus) if gpus else 0)

# =================  ADVANCED DATA AUGMENTATION  =================
print("📊 Creating advanced data augmentation...")

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    # Rotation
    rotation_range=20,          # More rotation
    # Translation  
    width_shift_range=0.1,      # More shifting
    height_shift_range=0.1,
    # Zoom
    zoom_range=0.15,            # More zoom
    # Flip
    horizontal_flip=True,
    vertical_flip=True,         # Vertical flip suitable for astronomy
    # Brightness
    brightness_range=[0.8, 1.2], # Wider brightness range
    # Shear
    shear_range=0.1,            # Light shear transformation
    # Fill mode
    fill_mode='nearest'
)

val_gen = ImageDataGenerator(preprocessing_function=preprocess_input)

# =================  DATA LOADING  =================
print("📁 Loading dataset...")

train_ds = train_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    seed=SEED
)

val_ds = val_gen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class names
class_names = list(train_ds.class_indices.keys())
os.makedirs(os.path.dirname(CLASS_NAMES_PATH), exist_ok=True)
with open(CLASS_NAMES_PATH, 'w') as f:
    json.dump(class_names, f)

print(f"📋 Classes: {class_names}")
print(f"📊 Training samples: {train_ds.samples}")
print(f"📊 Validation samples: {val_ds.samples}")

# =================  CLASS WEIGHTS COMPUTATION  =================
print("⚖️ Computing class weights...")

# Get class distribution from training set
train_labels = train_ds.labels
unique_labels = np.unique(train_labels)
class_weights_array = compute_class_weight(
    'balanced',
    classes=unique_labels,
    y=train_labels
)

class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
print(f"⚖️ Class weights: {class_weights}")

# =================  IMPROVED MODEL ARCHITECTURE  =================
print("🏗️ Creating improved model architecture...")

def create_improved_model():
    # Input layer
    inputs = Input(shape=(*IMAGE_SIZE, 3))
    
    # MobileNetV2 base model
    base_model = MobileNetV2(
        input_tensor=inputs,
        weights='imagenet',
        include_top=False,
        alpha=1.0  # Full model
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    
    # Dropout for regularization
    x = Dropout(0.3, name='dropout_1')(x)
    
    # Dense layer with batch normalization
    x = Dense(256, activation='relu', name='dense_256')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dropout(0.4, name='dropout_2')(x)
    
    # Another dense layer
    x = Dense(128, activation='relu', name='dense_128')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dropout(0.3, name='dropout_3')(x)
    
    # Output layer
    predictions = Dense(len(class_names), activation='softmax', name='predictions')(x)
    
    model = Model(inputs=inputs, outputs=predictions)
    return model, base_model

model, base_model = create_improved_model()

# =================  COMPILATION  =================
print("⚙️ Compiling model...")

# Improved loss function with label smoothing
loss_fn = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=LABEL_SMOOTH,
    name='categorical_crossentropy'
)

# Metrics
metrics = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy')
]

model.compile(
    optimizer=Adam(learning_rate=BASE_LR),
    loss=loss_fn,
    metrics=metrics
)

print(f"📐 Model summary:")
model.summary()

# =================  CALLBACKS  =================
print("📞 Callbacks hazırlanıyor...")

# Model save paths
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

callbacks = [
    ModelCheckpoint(
        KERAS_SAVE_PATH,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1,
        save_format='keras'
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=7,  # Daha fazla patience
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    CSVLogger(IMPROVED_LOG_PATH, append=False)
]

# =================  STAGE 1: FROZEN BASE TRAINING  =================
print("\n" + "="*60)
print("🚀 STAGE 1: Frozen Base Training")
print("="*60)

history_stage1 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE1,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weights,  # Important: use class weights
    verbose=1
)

# =================  STAGE 2: FINE-TUNING  =================
print("\n" + "="*60)
print("🔥 STAGE 2: Fine-tuning")
print("="*60)

# Unfreeze the base model
base_model.trainable = True

# Fine-tune from the top layers of the pre-trained model
fine_tune_at = len(base_model.layers) // 2  # Fine-tune the top half

# Freeze the bottom layers
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=BASE_LR/10),  # Lower learning rate
    loss=loss_fn,
    metrics=metrics
)

print(f"🔧 Fine-tuning {len(base_model.layers) - fine_tune_at} layers")

# Continue training
history_stage2 = model.fit(
    train_ds,
    epochs=EPOCHS_STAGE2,
    validation_data=val_ds,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1,
    initial_epoch=EPOCHS_STAGE1
)

# =================  SAVE FINAL MODEL  =================
print("\n💾 Saving final model...")

# Save in H5 format as well
model.save(MODEL_SAVE_PATH, save_format='h5')
print(f"✅ Model saved: {MODEL_SAVE_PATH}")
print(f"✅ Model saved: {KERAS_SAVE_PATH}")

# =================  SAVE TRAINING HISTORY  =================
print("📊 Saving training history...")

# Combine histories
combined_history = {}
for key in history_stage1.history.keys():
    combined_history[key] = history_stage1.history[key] + history_stage2.history[key]

# Save history
np.save(IMPROVED_HISTORY_PATH, combined_history)
print(f"✅ History saved: {IMPROVED_HISTORY_PATH}")

# =================  PLOT TRAINING RESULTS  =================
print("📈 Creating training graphs...")

def plot_training_results(history, save_path_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history['accuracy'], label='Training Accuracy')
    axes[0, 0].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history['loss'], label='Training Loss')
    axes[0, 1].plot(history['val_loss'], label='Validation Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history['precision'], label='Training Precision')
    axes[1, 0].plot(history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history['recall'], label='Training Recall')
    axes[1, 1].plot(history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path_prefix}_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_training_results(combined_history, 'models/improved')
print("✅ Graphs saved: models/improved_training_results.png")

# =================  FINAL SUMMARY  =================
print("\n" + "="*60)
print("🎉 IMPROVED MODEL TRAINING COMPLETED!")
print("="*60)
print(f"📁 Model file: {MODEL_SAVE_PATH}")
print(f"📁 Keras model: {KERAS_SAVE_PATH}")
print(f"📁 Training log: {IMPROVED_LOG_PATH}")
print(f"📁 History: {IMPROVED_HISTORY_PATH}")
print("📁 Graphs: models/improved_training_results.png")

# Final metrics
final_val_acc = combined_history['val_accuracy'][-1]
final_val_loss = combined_history['val_loss'][-1]
best_val_acc = max(combined_history['val_accuracy'])

print(f"\n📊 FINAL METRICS:")
print(f"🎯 Final Validation Accuracy: {final_val_acc:.4f}")
print(f"📉 Final Validation Loss: {final_val_loss:.4f}")
print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f}")

print("\n✨ Now you can make predictions with the improved model!")
print("🔧 Usage: python scripts/weights_predict.py <image_path>")
print("   (Update weights_predict.py for the improved model)")
