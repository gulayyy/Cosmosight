import numpy as np
import matplotlib.pyplot as plt
import os, sys

# Add parent folder to sys.path (config.py is there)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# History path for improved model
IMPROVED_HISTORY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "improved_training_history.npy")

# Load improved model training history
print("📊 Loading improved model training history...")
history = np.load(IMPROVED_HISTORY_PATH, allow_pickle=True).item()
print(f"✅ History loaded - {len(history['loss'])} epochs")

# Advanced Loss Graph
plt.figure(figsize=(12, 6))
plt.plot(history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
plt.title('Improved Model - Loss Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Mark best loss
min_val_loss_epoch = np.argmin(history['val_loss'])
min_val_loss = np.min(history['val_loss'])
plt.plot(min_val_loss_epoch, min_val_loss, 'ro', markersize=8)
plt.annotate(f'Best Val Loss: {min_val_loss:.4f}\nEpoch: {min_val_loss_epoch+1}', 
             xy=(min_val_loss_epoch, min_val_loss), 
             xytext=(min_val_loss_epoch+2, min_val_loss+0.01),
             fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

# Get full path to models folder
models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
loss_graph_path = os.path.join(models_dir, "improved_loss_graph.png")

plt.tight_layout()
plt.savefig(loss_graph_path, dpi=300, bbox_inches='tight')
print(f"💾 Loss graph saved: {loss_graph_path}")
plt.show()

# Advanced Accuracy Graph
plt.figure(figsize=(12, 6))
plt.plot(history['accuracy'], label='Training Accuracy', linewidth=2, color='#27ae60')
plt.plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#f39c12')
plt.title('Improved Model - Accuracy Graph', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Mark best accuracy
max_val_acc_epoch = np.argmax(history['val_accuracy'])
max_val_acc = np.max(history['val_accuracy'])
plt.plot(max_val_acc_epoch, max_val_acc, 'go', markersize=8)
plt.annotate(f'Best Val Acc: {max_val_acc:.4f}\nEpoch: {max_val_acc_epoch+1}', 
             xy=(max_val_acc_epoch, max_val_acc), 
             xytext=(max_val_acc_epoch+2, max_val_acc-0.02),
             fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

# Accuracy graph path
accuracy_graph_path = os.path.join(models_dir, "improved_accuracy_graph.png")

plt.tight_layout()
plt.savefig(accuracy_graph_path, dpi=300, bbox_inches='tight')
print(f"💾 Accuracy graph saved: {accuracy_graph_path}")
plt.show()

# Final metrics summary
print("\n📈 Improved Model - Final Metrics:")
print(f"🎯 Final Training Accuracy: {history['accuracy'][-1]:.4f}")
print(f"🎯 Final Validation Accuracy: {history['val_accuracy'][-1]:.4f}")
print(f"📉 Final Training Loss: {history['loss'][-1]:.4f}")
print(f"📉 Final Validation Loss: {history['val_loss'][-1]:.4f}")
print(f"⭐ Best Validation Accuracy: {max_val_acc:.4f} (Epoch {max_val_acc_epoch+1})")
print(f"⭐ Best Validation Loss: {min_val_loss:.4f} (Epoch {min_val_loss_epoch+1})")
