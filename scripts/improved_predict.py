import os
import sys
import numpy as np
from PIL import Image
import json

# TensorFlow import and setup
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Set TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def load_improved_model():
    """Load improved model"""
    try:
        # First try improved model
        keras_path = "models/astro_model_improved.keras"
        h5_path = "models/astro_model_improved.h5"
        
        model_path = None
        if os.path.exists(keras_path):
            model_path = keras_path
            print(f"✅ Improved Keras model found: {keras_path}")
        elif os.path.exists(h5_path):
            model_path = h5_path
            print(f"✅ Improved H5 model found: {h5_path}")
        else:
            # Fallback to original model
            print("⚠️ Improved model not found, trying original model...")
            original_keras = "models/astro_model.keras"
            original_h5 = "models/astro_model.h5"
            
            if os.path.exists(original_keras):
                model_path = original_keras
                print(f"📦 Original Keras model found: {original_keras}")
            elif os.path.exists(original_h5):
                model_path = original_h5
                print(f"📦 Original H5 model found: {original_h5}")
            else:
                print("❌ No model files found!")
                return None
        
        print("🔄 Loading model (this may take a moment)...")
        
        # Model loading
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model loaded successfully!")
            return model
        except Exception as e:
            print(f"❌ Model loading error: {str(e)[:200]}...")
            print("🔄 Trying weights extraction method...")
            return None
            
    except Exception as e:
        print(f"❌ General model loading error: {str(e)[:200]}...")
        return None

def preprocess_image_for_prediction(image_path):
    """Prepare image for prediction"""
    try:
        # Open image and convert to RGB
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize (224x224)
        img = img.resize((224, 224))
        
        # Convert to NumPy array
        img_array = np.array(img, dtype=np.float32)
        
        # Add batch dimension
        img_batch = np.expand_dims(img_array, axis=0)
        
        # MobileNetV2 preprocessing ([-1, 1] range)
        img_preprocessed = preprocess_input(img_batch)
        
        return img_preprocessed
        
    except Exception as e:
        print(f"❌ Image preprocessing error: {e}")
        return None

def predict_with_confidence_analysis(model, image_path, class_names):
    """Advanced prediction analysis"""
    try:
        print(f"🔍 Image analysis: {os.path.basename(image_path)}")
        
        # Preprocess image
        img_processed = preprocess_image_for_prediction(image_path)
        if img_processed is None:
            return None
        
        print("🧠 Model making prediction...")
        
        # Make prediction
        predictions = model.predict(img_processed, verbose=0)
        probabilities = predictions[0]  # Get first (and only) batch element
        
        # Analyze results
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_class = class_names[predicted_idx]
        
        # Confidence level analysis
        if confidence >= 0.8:
            confidence_level = "Very High 🟢"
        elif confidence >= 0.6:
            confidence_level = "High 🟡"
        elif confidence >= 0.4:
            confidence_level = "Medium 🟠"
        else:
            confidence_level = "Low 🔴"
        
        # Second highest score
        sorted_indices = np.argsort(probabilities)[::-1]
        second_idx = sorted_indices[1]
        second_confidence = probabilities[second_idx]
        second_class = class_names[second_idx]
        
        # Margin (difference)
        margin = confidence - second_confidence
        
        # Show results
        print("\n" + "="*70)
        print("🎯 IMPROVED MODEL PREDICTION")
        print("="*70)
        print(f"📸 Image: {os.path.basename(image_path)}")
        print(f"🏷️  Most Likely Class: {predicted_class}")
        print(f"📊 Confidence Score: {confidence:.2%}")
        print(f"🎚️  Confidence Level: {confidence_level}")
        print(f"📏 Margin from second class: {margin:.2%}")
        print("="*70)
        
        print(f"\n📈 Detailed Class Scores:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            if i == predicted_idx:
                marker = "🎯"
                status = "(PREDICTION)"
            elif i == second_idx:
                marker = "🥈"
                status = "(SECOND)"
            else:
                marker = "  "
                status = ""
            
            # Progress bar
            bar_length = 20
            filled_length = int(bar_length * prob)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            print(f"{marker} {class_name:8s}: {prob:6.2%} |{bar}| {status}")
        
        # Reliability analysis
        print(f"\n🔍 Reliability Analysis:")
        if confidence >= 0.7 and margin >= 0.3:
            reliability = "Very reliable prediction ✅"
        elif confidence >= 0.5 and margin >= 0.2:
            reliability = "Reliable prediction ☑️"
        elif confidence >= 0.4:
            reliability = "Medium reliability ⚠️"
        else:
            reliability = "Low reliability - Manual check recommended ❌"
        
        print(f"🔮 {reliability}")
        
        # Alternative predictions
        if margin < 0.2:
            print(f"💡 Note: {second_class} class also has close score of {second_confidence:.1%}")
            print(f"   Manual verification is recommended.")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'margin': margin,
            'reliability': reliability
        }
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def batch_predict(model, image_paths, class_names):
    """Prediction for multiple images"""
    results = []
    
    print(f"\n🔄 Processing {len(image_paths)} images...")
    
    for i, image_path in enumerate(image_paths):
        print(f"\n[{i+1}/{len(image_paths)}]", end=" ")
        result = predict_with_confidence_analysis(model, image_path, class_names)
        if result:
            results.append({
                'image': os.path.basename(image_path),
                'path': image_path,
                **result
            })
    
    return results

def main():
    if len(sys.argv) < 2:
        print("❌ Usage:")
        print("   Single image: python improved_predict.py <image_path>")
        print("   Multiple images: python improved_predict.py <image1> <image2> ...")
        print("   Example: python improved_predict.py data/split_dataset/test/Galaxy/test-galaxy.jpg")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    # Check file existence
    valid_paths = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            valid_paths.append(image_path)
        else:
            print(f"⚠️ File not found: {image_path}")
    
    if not valid_paths:
        print("❌ No valid image files found!")
        sys.exit(1)
    
    # Load class names
    try:
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
        print(f"✅ Classes loaded: {class_names}")
    except Exception as e:
        print(f"❌ Could not load class names: {e}")
        sys.exit(1)
    
    # Load model
    print("🔄 Loading model...")
    model = load_improved_model()
    if model is None:
        print("❌ Could not load model!")
        sys.exit(1)
    
    print(f"✅ Model loaded successfully!")
    print(f"📐 Model input shape: {model.input_shape}")
    print(f"📐 Model output shape: {model.output_shape}")
    
    # Make prediction(s)
    if len(valid_paths) == 1:
        # Single image
        predict_with_confidence_analysis(model, valid_paths[0], class_names)
    else:
        # Multiple images
        results = batch_predict(model, valid_paths, class_names)
        
        # Summary
        print("\n" + "="*70)
        print("📊 BATCH PREDICTION SUMMARY")
        print("="*70)
        
        for result in results:
            reliability_emoji = "✅" if "Very reliable" in result['reliability'] else \
                              "☑️" if "Reliable" in result['reliability'] else \
                              "⚠️" if "Moderate" in result['reliability'] else "❌"
            
            print(f"{reliability_emoji} {result['image']:30s} -> {result['predicted_class']:8s} ({result['confidence']:5.1%})")

if __name__ == "__main__":
    main()
