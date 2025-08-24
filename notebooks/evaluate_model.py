import os, sys, json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # MobileNetV2 preprocess

# ================== Settings ==================
USE_CPU_ONLY     = False                 # set to True if you want to run with CPU only
IMG_SIZE         = (224, 224)
BATCH_SIZE       = 8
PREPROCESS_MODE  = "mobilenet"           # "mobilenet" (preprocess_input) || "rescale"
LABEL_SMOOTH     = 0.05                  # same value used in training
# =============================================

if USE_CPU_ONLY:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Add project root to path (for config.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create absolute paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(project_root, "data", "split_dataset", "test")
CLASS_NAMES_PATH = os.path.join(project_root, "models", "class_names.json")

# Path for improved model
IMPROVED_MODEL_PATH = os.path.join(project_root, "models", "astro_model_improved.keras")

# ---- GPU bellek yönetimi: ihtiyaç kadar bellek ayır ----
gpus = tf.config.list_physical_devices('GPU')
if gpus and not USE_CPU_ONLY:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

print(f"\nImproved Model path: {IMPROVED_MODEL_PATH}")
print(f"Test folder: {TEST_DIR}")
print(f"Preprocessing mode: {PREPROCESS_MODE}\n")

# ---- Load improved model and class names ----
print("🔄 Loading improved model...")
model = tf.keras.models.load_model(IMPROVED_MODEL_PATH)
with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)
print("✅ Model loaded!")
print("CLASS_NAMES.json:", class_names)

# ---- PATCH: recompile with named metrics ----
try:
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTH),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ]
    )
except Exception:
    pass  # might already be compiled in some cases

# ---- Dataset builder (function to rebuild) ----
def make_test_ds():
    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="categorical",
        class_names=class_names  # fix ordering
    )
    if PREPROCESS_MODE == "mobilenet":
        map_fn = lambda x, y: (preprocess_input(tf.cast(x, tf.float32)), y)
    elif PREPROCESS_MODE == "rescale":
        map_fn = lambda x, y: (tf.cast(x, tf.float32) / 255.0, y)
    else:
        raise ValueError("PREPROCESS_MODE must be 'mobilenet' or 'rescale'.")
    ds = ds.map(map_fn, num_parallel_calls=1).prefetch(1)
    return ds

# ---- Evaluation ----
test_ds_eval = make_test_ds()
results = model.evaluate(test_ds_eval, verbose=1)
metric_names = model.metrics_names  # should now be ['loss','accuracy','precision','recall']
metrics = dict(zip(metric_names, [float(x) for x in results]))

print("\nEval metrics:", {k: round(metrics[k], 6) for k in metric_names})
print(f"Test Loss: {metrics['loss']:.4f}  |  "
      f"Test Acc: {metrics.get('accuracy', float('nan')):.4f}  |  "
      f"Precision: {metrics.get('precision', float('nan')):.4f}  |  "
      f"Recall: {metrics.get('recall', float('nan')):.4f}")

# ---- True and predicted labels (with separate datasets) ----
test_ds_true = make_test_ds()
y_true = np.argmax(
    np.concatenate([y.numpy() for _, y in test_ds_true], axis=0),
    axis=1
)

test_ds_pred = make_test_ds()
pred_raw = model.predict(test_ds_pred, verbose=1)
y_pred = np.argmax(pred_raw, axis=1)

# --- Diagnostic: Prediction class distribution ---
pred_counts = np.bincount(y_pred, minlength=len(class_names))
print("Pred class counts (in order", class_names, "):", pred_counts.tolist())

# ---- Confusion Matrix (counts) ----
print("\n📊 Creating confusion matrix visualizations...")
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(models_dir, exist_ok=True)

cm = confusion_matrix(y_true, y_pred)
fig = plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap='Blues')
plt.title("🔢 Improved Model - Confusion Matrix (Counts)", fontsize=14, fontweight='bold', pad=20)
plt.colorbar()

# Print confusion matrix values
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12, fontweight='bold')

tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=12)
plt.yticks(tick_marks, class_names, fontsize=12)
plt.tight_layout()
plt.ylabel("True Label", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')

confusion_counts_path = os.path.join(models_dir, "improved_confusion_matrix_counts.png")
plt.savefig(confusion_counts_path, dpi=300, bbox_inches='tight')
print(f"💾 Confusion matrix (counts) saved: {confusion_counts_path}")
plt.show()
plt.close(fig)

# ---- Confusion Matrix (normalized) ----
cm_norm = cm.astype(np.float32) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
fig = plt.figure(figsize=(10, 8))
plt.imshow(cm_norm, interpolation='nearest', cmap='Oranges')
plt.title("📊 Improved Model - Confusion Matrix (Normalized)", fontsize=14, fontweight='bold', pad=20)
plt.colorbar()

# Print normalized values
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, f'{cm_norm[i, j]:.3f}', ha='center', va='center', fontsize=12, fontweight='bold')

plt.xticks(tick_marks, class_names, rotation=45, ha='right', fontsize=12)
plt.yticks(tick_marks, class_names, fontsize=12)
plt.tight_layout()
plt.ylabel("True Label", fontsize=12, fontweight='bold')
plt.xlabel("Predicted Label", fontsize=12, fontweight='bold')

confusion_norm_path = os.path.join(models_dir, "improved_confusion_matrix_normalized.png")
plt.savefig(confusion_norm_path, dpi=300, bbox_inches='tight')
print(f"💾 Confusion matrix (normalized) saved: {confusion_norm_path}")
plt.show()
plt.close(fig)

# ---- Classification report ----
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n📈 Improved Model - Classification Report:")
print("="*60)
print(report)

# Save report to file
report_path = os.path.join(models_dir, "improved_classification_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("IMPROVED MODEL - EVALUATION REPORT\n")
    f.write("="*50 + "\n\n")
    f.write(f"Model: {IMPROVED_MODEL_PATH}\n")
    f.write(f"Preprocess: {PREPROCESS_MODE}\n")
    f.write(f"Test Dataset: {TEST_DIR}\n\n")
    f.write("METRICS:\n")
    for k in metric_names:
        f.write(f"{k}: {metrics[k]:.6f}\n")
    f.write(f"\nPred class counts: {json.dumps(pred_counts.tolist())}\n\n")
    f.write("CONFUSION MATRIX (COUNTS):\n")
    f.write(f"Classes: {class_names}\n")
    for i, row in enumerate(cm):
        f.write(f"{class_names[i]}: {row.tolist()}\n")
    f.write(f"\nCLASSIFICATION REPORT:\n")
    f.write(report)

print(f"💾 Classification report saved: {report_path}")
print(f"\n✅ All evaluation visualizations and reports created!")
print(f"📁 Files: {models_dir}")
