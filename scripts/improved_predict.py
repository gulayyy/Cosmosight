import os
import sys
import numpy as np
from PIL import Image
import json

# TensorFlow import and setup
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# TensorFlow log seviyesini kur
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def load_improved_model():
    """Geliştirilmiş modeli yükle"""
    try:
        # Önce improved model'i dene
        keras_path = "models/astro_model_improved.keras"
        h5_path = "models/astro_model_improved.h5"
        
        model_path = None
        if os.path.exists(keras_path):
            model_path = keras_path
            print(f"✅ Improved Keras modeli bulundu: {keras_path}")
        elif os.path.exists(h5_path):
            model_path = h5_path
            print(f"✅ Improved H5 modeli bulundu: {h5_path}")
        else:
            # Fallback to original model
            print("⚠️ Improved model bulunamadı, orijinal model denenecek...")
            original_keras = "models/astro_model.keras"
            original_h5 = "models/astro_model.h5"
            
            if os.path.exists(original_keras):
                model_path = original_keras
                print(f"📦 Original Keras modeli bulundu: {original_keras}")
            elif os.path.exists(original_h5):
                model_path = original_h5
                print(f"📦 Original H5 modeli bulundu: {original_h5}")
            else:
                print("❌ Hiçbir model dosyası bulunamadı!")
                return None
        
        print("🔄 Model yükleniyor (bu biraz sürebilir)...")
        
        # Model yükleme
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Model başarıyla yüklendi!")
            return model
        except Exception as e:
            print(f"❌ Model yükleme hatası: {str(e)[:200]}...")
            print("🔄 Weights extraction metoduyla deneniyor...")
            return None
            
    except Exception as e:
        print(f"❌ Genel model yükleme hatası: {str(e)[:200]}...")
        return None

def preprocess_image_for_prediction(image_path):
    """Resmi tahmin için hazırla"""
    try:
        # Resmi aç ve RGB'ye çevir
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Boyutlandır (224x224)
        img = img.resize((224, 224))
        
        # NumPy array'e çevir
        img_array = np.array(img, dtype=np.float32)
        
        # Batch dimension ekle
        img_batch = np.expand_dims(img_array, axis=0)
        
        # MobileNetV2 için preprocessing ([-1, 1] aralığı)
        img_preprocessed = preprocess_input(img_batch)
        
        return img_preprocessed
        
    except Exception as e:
        print(f"❌ Resim preprocessing hatası: {e}")
        return None

def predict_with_confidence_analysis(model, image_path, class_names):
    """Gelişmiş tahmin analizi"""
    try:
        print(f"🔍 Resim analizi: {os.path.basename(image_path)}")
        
        # Resmi preprocess et
        img_processed = preprocess_image_for_prediction(image_path)
        if img_processed is None:
            return None
        
        print("🧠 Model tahmin yapıyor...")
        
        # Tahmin yap
        predictions = model.predict(img_processed, verbose=0)
        probabilities = predictions[0]  # İlk (ve tek) batch elementini al
        
        # Sonuçları analiz et
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        predicted_class = class_names[predicted_idx]
        
        # Confidence level analizi
        if confidence >= 0.8:
            confidence_level = "Çok Yüksek 🟢"
        elif confidence >= 0.6:
            confidence_level = "Yüksek 🟡"
        elif confidence >= 0.4:
            confidence_level = "Orta 🟠"
        else:
            confidence_level = "Düşük 🔴"
        
        # İkinci en yüksek skor
        sorted_indices = np.argsort(probabilities)[::-1]
        second_idx = sorted_indices[1]
        second_confidence = probabilities[second_idx]
        second_class = class_names[second_idx]
        
        # Margin (fark)
        margin = confidence - second_confidence
        
        # Sonuçları göster
        print("\n" + "="*70)
        print("🎯 GELİŞTİRİLMİŞ MODEL TAHMİNİ")
        print("="*70)
        print(f"📸 Resim: {os.path.basename(image_path)}")
        print(f"🏷️  En Olası Sınıf: {predicted_class}")
        print(f"📊 Güven Skoru: {confidence:.2%}")
        print(f"🎚️  Güven Seviyesi: {confidence_level}")
        print(f"📏 İkinci sınıftan fark: {margin:.2%}")
        print("="*70)
        
        print(f"\n📈 Detaylı Sınıf Skorları:")
        for i, (class_name, prob) in enumerate(zip(class_names, probabilities)):
            if i == predicted_idx:
                marker = "🎯"
                status = "(TAHMİN)"
            elif i == second_idx:
                marker = "🥈"
                status = "(İKİNCİ)"
            else:
                marker = "  "
                status = ""
            
            # Progress bar
            bar_length = 20
            filled_length = int(bar_length * prob)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            print(f"{marker} {class_name:8s}: {prob:6.2%} |{bar}| {status}")
        
        # Güvenilirlik analizi
        print(f"\n🔍 Güvenilirlik Analizi:")
        if confidence >= 0.7 and margin >= 0.3:
            reliability = "Çok güvenilir tahmin ✅"
        elif confidence >= 0.5 and margin >= 0.2:
            reliability = "Güvenilir tahmin ☑️"
        elif confidence >= 0.4:
            reliability = "Orta güvenilirlik ⚠️"
        else:
            reliability = "Düşük güvenilirlik - Manuel kontrol önerilir ❌"
        
        print(f"🔮 {reliability}")
        
        # Alternatif tahminler
        if margin < 0.2:
            print(f"💡 Not: {second_class} sınıfı da {second_confidence:.1%} ile yakın skorlu")
            print(f"   Manuel kontrol yapmanız önerilir.")
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'margin': margin,
            'reliability': reliability
        }
        
    except Exception as e:
        print(f"❌ Tahmin hatası: {e}")
        return None

def batch_predict(model, image_paths, class_names):
    """Birden fazla resim için tahmin"""
    results = []
    
    print(f"\n🔄 {len(image_paths)} resim işleniyor...")
    
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
        print("❌ Kullanım:")
        print("   Tek resim: python improved_predict.py <resim_yolu>")
        print("   Çoklu resim: python improved_predict.py <resim1> <resim2> ...")
        print("   Örnek: python improved_predict.py data/split_dataset/test/Galaxy/test-galaksi.jpg")
        sys.exit(1)
    
    image_paths = sys.argv[1:]
    
    # Dosya varlığını kontrol et
    valid_paths = []
    for image_path in image_paths:
        if os.path.exists(image_path):
            valid_paths.append(image_path)
        else:
            print(f"⚠️ Dosya bulunamadı: {image_path}")
    
    if not valid_paths:
        print("❌ Geçerli resim dosyası bulunamadı!")
        sys.exit(1)
    
    # Class names'i yükle
    try:
        with open('models/class_names.json', 'r') as f:
            class_names = json.load(f)
        print(f"✅ Sınıflar yüklendi: {class_names}")
    except Exception as e:
        print(f"❌ Class names yüklenemedi: {e}")
        sys.exit(1)
    
    # Modeli yükle
    print("🔄 Model yükleniyor...")
    model = load_improved_model()
    if model is None:
        print("❌ Model yüklenemedi!")
        sys.exit(1)
    
    print(f"✅ Model başarıyla yüklendi!")
    print(f"📐 Model input shape: {model.input_shape}")
    print(f"📐 Model output shape: {model.output_shape}")
    
    # Tahmin(ler) yap
    if len(valid_paths) == 1:
        # Tek resim
        predict_with_confidence_analysis(model, valid_paths[0], class_names)
    else:
        # Çoklu resim
        results = batch_predict(model, valid_paths, class_names)
        
        # Özet
        print("\n" + "="*70)
        print("📊 TOPLU TAHMİN ÖZETİ")
        print("="*70)
        
        for result in results:
            reliability_emoji = "✅" if "Çok güvenilir" in result['reliability'] else \
                              "☑️" if "Güvenilir" in result['reliability'] else \
                              "⚠️" if "Orta" in result['reliability'] else "❌"
            
            print(f"{reliability_emoji} {result['image']:30s} -> {result['predicted_class']:8s} ({result['confidence']:5.1%})")

if __name__ == "__main__":
    main()
