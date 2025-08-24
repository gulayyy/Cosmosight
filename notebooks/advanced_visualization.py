#!/usr/bin/env python3
"""
Advanced Training Visualization Script
Creates training graphs for the Astronomy Classification Model
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style
import os
import sys
from datetime import datetime

# Matplotlib style ayarları
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

def load_training_history():
    """Load improved model training history"""
    # Find project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    history_path = os.path.join(project_root, "models", "improved_training_history.npy")
    
    if not os.path.exists(history_path):
        raise FileNotFoundError(f"Training history not found: {history_path}")
    
    print("📊 Loading improved model training history...")
    history = np.load(history_path, allow_pickle=True).item()
    print(f"✅ History loaded - {len(history['loss'])} epochs")
    
    return history, project_root

def create_loss_graph(history, save_path):
    """Create loss graph"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss curves
    plt.plot(epochs, history['loss'], 'b-', linewidth=2.5, label='Training Loss', alpha=0.8)
    plt.plot(epochs, history['val_loss'], 'r-', linewidth=2.5, label='Validation Loss', alpha=0.8)
    
    # Mark best loss point
    min_val_loss_epoch = np.argmin(history['val_loss']) + 1
    min_val_loss = np.min(history['val_loss'])
    
    plt.plot(min_val_loss_epoch, min_val_loss, 'ro', markersize=10, label=f'Best Val Loss (Epoch {min_val_loss_epoch})')
    
    # Annotations
    plt.annotate(f'Best: {min_val_loss:.4f}', 
                xy=(min_val_loss_epoch, min_val_loss), 
                xytext=(min_val_loss_epoch + 2, min_val_loss + 0.05),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="yellow", alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Styling
    plt.title('🔥 Improved Model - Loss Curve', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Loss', fontsize=13, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add final loss values
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    plt.text(0.02, 0.98, f'Final Training Loss: {final_train_loss:.4f}\nFinal Validation Loss: {final_val_loss:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Loss graph saved: {save_path}")
    
    return plt.gcf()

def create_accuracy_graph(history, save_path):
    """Create accuracy graph"""
    plt.figure(figsize=(12, 8))
    
    epochs = range(1, len(history['accuracy']) + 1)
    
    # Accuracy curves
    plt.plot(epochs, history['accuracy'], 'g-', linewidth=2.5, label='Training Accuracy', alpha=0.8)
    plt.plot(epochs, history['val_accuracy'], 'orange', linewidth=2.5, label='Validation Accuracy', alpha=0.8)
    
    # Mark best accuracy point
    max_val_acc_epoch = np.argmax(history['val_accuracy']) + 1
    max_val_acc = np.max(history['val_accuracy'])
    
    plt.plot(max_val_acc_epoch, max_val_acc, 'go', markersize=10, label=f'Best Val Acc (Epoch {max_val_acc_epoch})')
    
    # Annotations
    plt.annotate(f'Best: {max_val_acc:.4f}', 
                xy=(max_val_acc_epoch, max_val_acc), 
                xytext=(max_val_acc_epoch + 2, max_val_acc - 0.02),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Styling
    plt.title('🎯 Improved Model - Accuracy Curve', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontsize=13, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Y axis limits
    plt.ylim(0.5, 1.02)
    
    # Add final accuracy values
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    
    plt.text(0.02, 0.02, f'Final Training Acc: {final_train_acc:.4f}\nFinal Validation Acc: {final_val_acc:.4f}', 
             transform=plt.gca().transAxes, fontsize=10, fontweight='bold',
             verticalalignment='bottom', bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Accuracy graph saved: {save_path}")
    
    return plt.gcf()

def create_combined_graph(history, save_path):
    """Combined graph showing both loss and accuracy"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss subplot
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2, label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    ax1.set_title('Loss Curve', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy subplot
    ax2.plot(epochs, history['accuracy'], 'g-', linewidth=2, label='Training Accuracy')
    ax2.plot(epochs, history['val_accuracy'], 'orange', linewidth=2, label='Validation Accuracy')
    ax2.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.02)
    
    plt.suptitle('🚀 Improved Model - Training Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"💾 Combined graph saved: {save_path}")
    
    return fig

def print_training_summary(history):
    """Print training summary"""
    print("\n" + "="*60)
    print("📈 IMPROVED MODEL - TRAINING SUMMARY")
    print("="*60)
    
    # Final metrics
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    # Best metrics
    best_val_acc = np.max(history['val_accuracy'])
    best_val_acc_epoch = np.argmax(history['val_accuracy']) + 1
    best_val_loss = np.min(history['val_loss'])
    best_val_loss_epoch = np.argmin(history['val_loss']) + 1
    
    print(f"🎯 Final Metrics:")
    print(f"   Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
    print(f"   Training Loss: {final_train_loss:.4f}")
    print(f"   Validation Loss: {final_val_loss:.4f}")
    
    print(f"\n⭐ Best Performance:")
    print(f"   Best Val Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%) - Epoch {best_val_acc_epoch}")
    print(f"   Best Val Loss: {best_val_loss:.4f} - Epoch {best_val_loss_epoch}")
    
    # Overfitting analysis
    overfit_score = final_train_acc - final_val_acc
    if overfit_score < 0.02:
        overfit_status = "✅ Well balanced"
    elif overfit_score < 0.05:
        overfit_status = "⚠️ Slight overfitting"
    else:
        overfit_status = "❌ Overfitting present"
    
    print(f"\n🔍 Model Status:")
    print(f"   Overfitting Analysis: {overfit_status}")
    print(f"   Accuracy Difference: {overfit_score:.4f}")
    print(f"   Total Epochs: {len(history['loss'])}")
    
    print("="*60)

def main():
    """Main function"""
    try:
        # Load history
        history, project_root = load_training_history()
        
        # Output directory
        models_dir = os.path.join(project_root, "models")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Graph file paths
        loss_path = os.path.join(models_dir, "improved_loss_graph.png")
        accuracy_path = os.path.join(models_dir, "improved_accuracy_graph.png")
        combined_path = os.path.join(models_dir, f"improved_training_summary_{timestamp}.png")
        
        print("\n🎨 Creating graphs...")
        
        # Create graphs
        loss_fig = create_loss_graph(history, loss_path)
        plt.show()
        
        accuracy_fig = create_accuracy_graph(history, accuracy_path)
        plt.show()
        
        combined_fig = create_combined_graph(history, combined_path)
        plt.show()
        
        # Print training summary
        print_training_summary(history)
        
        print(f"\n✅ All graphs created successfully!")
        print(f"📁 Graph files: {models_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
