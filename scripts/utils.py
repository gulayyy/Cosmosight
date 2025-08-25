import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import json
import matplotlib.pyplot as plt
from PIL import Image

def calculate_class_weights(generator):
    """Original class weights function"""
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(generator.classes),
        y=generator.classes
    )
    return dict(enumerate(class_weights))

def calculate_class_weights_from_dir(train_dir, class_names):
    """
    Calculate class weights from directory
    """
    class_counts = {}
    
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
        else:
            class_counts[class_name] = 0
    
    print("Class distribution:")
    total_samples = sum(class_counts.values())
    for class_name, count in class_counts.items():
        percentage = (count / total_samples) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Calculate class weights
    labels = []
    for i, class_name in enumerate(class_names):
        labels.extend([i] * class_counts[class_name])
    
    labels = np.array(labels)
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    print("Class weights:")
    for i, (class_name, weight) in enumerate(zip(class_names, class_weights)):
        print(f"  {class_name}: {weight:.3f}")
    
    return class_weight_dict

def analyze_dataset_quality(data_dir, sample_size=50):
    """
    Analyze dataset quality
    """
    print("🔍 Starting dataset quality analysis...")
    
    class_stats = {}
    
    for class_name in ['Galaxy', 'Nebula', 'Star']:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.exists(class_path):
            continue
            
        print(f"\n📂 {class_name} class analysis:")
        
        # File list
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        total_files = len(files)
        
        print(f"  📊 Total files: {total_files}")
        
        # Sample files for analysis
        sample_files = files[:min(sample_size, total_files)]
        
        sizes = []
        corrupted = 0
        
        for filename in sample_files:
            filepath = os.path.join(class_path, filename)
            try:
                with Image.open(filepath) as img:
                    sizes.append(img.size)
            except Exception:
                corrupted += 1
        
        if sizes:
            widths, heights = zip(*sizes)
            unique_sizes = set(sizes)
            
            print(f"  📐 Size variety: {len(unique_sizes)} different sizes")
            print(f"  📏 Average size: {np.mean(widths):.0f}x{np.mean(heights):.0f}")
            print(f"  ⚠️ Corrupted files: {corrupted}/{len(sample_files)}")
        
        class_stats[class_name] = {
            'total_files': total_files,
            'corrupted': corrupted,
            'sizes': sizes
        }
    
    return class_stats

def create_data_visualization(train_dir, save_path="models/dataset_analysis.png"):
    """
    Visualize dataset distribution
    """
    class_counts = {}
    class_names = ['Galaxy', 'Nebula', 'Star']
    
    for class_name in class_names:
        class_path = os.path.join(train_dir, class_name)
        if os.path.exists(class_path):
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[class_name] = count
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Bar chart
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax1.bar(classes, counts, color=colors)
    ax1.set_title('Class Distribution (Training Set)')
    ax1.set_ylabel('Number of Samples')
    ax1.set_xlabel('Classes')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=classes, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Proportions')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Dataset visualization saved: {save_path}")
    
    return class_counts
