"""
Improved model test script
Tests and compares with the original model
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_class_names():
    """Load class names"""
    class_names_path = "models/class_names.json"
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r') as f:
            return json.load(f)
    else:
        return ['Galaxy', 'Nebula', 'Star']

def preprocess_image(img_path):
    """Prepare image for model"""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_with_improved_model(img_path):
    """Make prediction with improved model"""
    print(f"🔬 Making prediction with improved model: {img_path}")
    
    # Load model
    try:
        model = load_model("models/astro_model_improved.keras")
        print("✅ Improved model loaded: astro_model_improved.keras")
    except:
        try:
            model = load_model("models/astro_model_improved.h5")
            print("✅ Improved model loaded: astro_model_improved.h5")
        except Exception as e:
            print(f"❌ Could not load improved model: {e}")
            return None
    
    # Class names
    class_names = load_class_names()
    
    # Prepare image
    if not os.path.exists(img_path):
        print(f"❌ Image not found: {img_path}")
        return None
    
    img_array = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    
    # Process results
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100
    
    print(f"\n🎯 IMPROVED MODEL PREDICTION:")
    print(f"📂 File: {os.path.basename(img_path)}")
    print(f"🏆 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"\n📈 All class probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {class_name}: {prob*100:.2f}%")
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {name: prob*100 for name, prob in zip(class_names, probabilities)}
    }

def predict_with_original_model(img_path):
    """Make prediction with original model"""
    print(f"\n🔬 Making prediction with original model: {img_path}")
    
    # Load model
    try:
        model = load_model("models/astro_model.keras")
        print("✅ Original model loaded: astro_model.keras")
    except:
        try:
            model = load_model("models/astro_model.h5")
            print("✅ Original model loaded: astro_model.h5")
        except Exception as e:
            print(f"❌ Could not load original model: {e}")
            return None
    
    # Class names
    class_names = load_class_names()
    
    # Prepare image
    img_array = preprocess_image(img_path)
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    probabilities = predictions[0]
    
    # Process results
    predicted_class_idx = np.argmax(probabilities)
    predicted_class = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx] * 100
    
    print(f"\n🎯 ORIGINAL MODEL PREDICTION:")
    print(f"📂 File: {os.path.basename(img_path)}")
    print(f"🏆 Prediction: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print(f"\n📈 All class probabilities:")
    for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
        print(f"  {class_name}: {prob*100:.2f}%")
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': {name: prob*100 for name, prob in zip(class_names, probabilities)}
    }

def compare_models(img_path):
    """Compare two models"""
    print("="*60)
    print("🚀 MODEL COMPARISON TEST")
    print("="*60)
    
    # Improved model
    improved_result = predict_with_improved_model(img_path)
    
    # Original model
    original_result = predict_with_original_model(img_path)
    
    # Comparison
    if improved_result and original_result:
        print(f"\n" + "="*60)
        print("📊 COMPARISON RESULTS")
        print("="*60)
        print(f"🔬 Improved Model: {improved_result['predicted_class']} ({improved_result['confidence']:.2f}%)")
        print(f"🔬 Original Model:  {original_result['predicted_class']} ({original_result['confidence']:.2f}%)")
        
        if improved_result['predicted_class'] == original_result['predicted_class']:
            print("✅ Both models predicted the same class!")
        else:
            print("⚠️ Models predicted different classes!")
        
        # Confidence comparison
        conf_diff = improved_result['confidence'] - original_result['confidence']
        if conf_diff > 5:
            print(f"📈 Improved model is {conf_diff:.2f}% more confident")
        elif conf_diff < -5:
            print(f"📉 Original model is {abs(conf_diff):.2f}% more confident")
        else:
            print("📊 Confidence levels are similar")

def test_batch_images():
    """Test multiple images"""
    test_images = []
    
    # Find test images
    for class_name in ['Galaxy', 'Nebula', 'Star']:
        class_dir = f"data/split_dataset/test/{class_name}"
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                # Take first 3 images
                for img in images[:3]:
                    test_images.append((os.path.join(class_dir, img), class_name))
    
    print(f"\n🧪 Found {len(test_images)} test images")
    
    # Test each image
    improved_correct = 0
    original_correct = 0
    total_tests = 0
    
    for img_path, true_class in test_images:
        print(f"\n{'='*80}")
        print(f"🧪 TEST #{total_tests + 1}: {os.path.basename(img_path)} (True: {true_class})")
        print(f"{'='*80}")
        
        # Improved model
        improved_result = predict_with_improved_model(img_path)
        
        # Original model  
        original_result = predict_with_original_model(img_path)
        
        if improved_result and original_result:
            total_tests += 1
            
            # Accuracy check
            if improved_result['predicted_class'] == true_class:
                improved_correct += 1
                print("✅ Improved model CORRECT prediction!")
            else:
                print("❌ Improved model INCORRECT prediction!")
            
            if original_result['predicted_class'] == true_class:
                original_correct += 1
                print("✅ Original model CORRECT prediction!")
            else:
                print("❌ Original model INCORRECT prediction!")
    
    # Batch results
    if total_tests > 0:
        print(f"\n{'='*80}")
        print("📊 BATCH TEST RESULTS")
        print(f"{'='*80}")
        print(f"🧪 Total tests: {total_tests}")
        print(f"🎯 Improved model accuracy: {improved_correct}/{total_tests} ({improved_correct/total_tests*100:.1f}%)")
        print(f"🎯 Original model accuracy: {original_correct}/{total_tests} ({original_correct/total_tests*100:.1f}%)")
        
        if improved_correct > original_correct:
            print("🏆 Improved model is MORE SUCCESSFUL!")
        elif original_correct > improved_correct:
            print("🤔 Original model is more successful (unexpected)")
        else:
            print("🤝 Both models have equal performance")

def main():
    if len(sys.argv) > 1:
        # Single image test
        img_path = sys.argv[1]
        compare_models(img_path)
    else:
        # Batch test
        test_batch_images()

if __name__ == "__main__":
    main()
