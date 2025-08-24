#!/bin/bash
# Ubuntu Test Runner Script

echo "🧪 Ubuntu Test Runner - Astronomi Modeli"
echo "========================================="

# Virtual environment check
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Virtual environment bulunamadı!"
    echo "💡 Önce setup script'ini çalıştırın: bash ubuntu_run_setup.sh"
    exit 1
fi

# Activate virtual environment
echo "🔧 Virtual environment aktifleştiriliyor..."
source .venv/bin/activate

# Check if models exist
echo ""
echo "📄 Model dosyaları kontrol ediliyor..."
if [ ! -f "models/astro_model_improved.keras" ]; then
    echo "❌ Geliştirilmiş model bulunamadı!"
    echo "💡 Önce modeli eğitin: python3 scripts/train_improved_model.py"
    exit 1
fi

if [ ! -f "models/class_names.json" ]; then
    echo "❌ Class names dosyası bulunamadı!"
    exit 1
fi

echo "✅ Model dosyaları hazır"

# Test options menu
echo ""
echo "🎯 Test seçenekleri:"
echo "1) Hızlı model testi (3 resim)"
echo "2) Detaylı model testi (9 resim)"
echo "3) Tek resim tahmin et"
echo "4) Batch test (tüm test seti)"
echo "5) Model karşılaştırması"
echo ""
read -p "Seçiminizi yapın (1-5): " choice

case $choice in
    1)
        echo "🚀 Hızlı test başlatılıyor..."
        python3 scripts/improved_predict.py "data/split_dataset/test/Galaxy/100263_4e77db06.jpg"
        python3 scripts/improved_predict.py "data/split_dataset/test/Nebula/10002074_large2x_27414cf3.jpeg"
        python3 scripts/improved_predict.py "data/split_dataset/test/Star/fitscut (10).jpeg"
        ;;
    2)
        echo "🔬 Detaylı test başlatılıyor..."
        python3 scripts/test_improved_model.py
        ;;
    3)
        echo "📁 Test klasörü içeriği:"
        find data/split_dataset/test -name "*.jpg" -o -name "*.jpeg" | head -10
        echo ""
        read -p "Resim yolunu girin: " image_path
        python3 scripts/improved_predict.py "$image_path"
        ;;
    4)
        echo "📊 Batch test başlatılıyor..."
        echo "Test klasöründeki tüm resimler test edilecek..."
        python3 -c "
import os
from scripts.improved_predict import predict_with_confidence_analysis, load_improved_model, load_class_names

model = load_improved_model()
class_names = load_class_names()

for class_name in ['Galaxy', 'Nebula', 'Star']:
    test_dir = f'data/split_dataset/test/{class_name}'
    if os.path.exists(test_dir):
        images = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))][:3]
        for img in images:
            img_path = os.path.join(test_dir, img)
            print(f'\n=== Testing {class_name}: {img} ===')
            predict_with_confidence_analysis(model, img_path, class_names)
"
        ;;
    5)
        echo "⚖️ Model karşılaştırması başlatılıyor..."
        if [ -f "models/astro_model.h5" ]; then
            python3 scripts/test_improved_model.py "data/split_dataset/test/Galaxy/100263_4e77db06.jpg"
        else
            echo "❌ Orijinal model bulunamadı, sadece geliştirilmiş model test edilecek"
            python3 scripts/improved_predict.py "data/split_dataset/test/Galaxy/100263_4e77db06.jpg"
        fi
        ;;
    *)
        echo "❌ Geçersiz seçim!"
        exit 1
        ;;
esac

echo ""
echo "✅ Test tamamlandı!"
