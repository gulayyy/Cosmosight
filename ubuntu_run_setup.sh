#!/bin/bash
# Ubuntu Terminal Setup Script for Astro Classification Project

echo "🐧 Ubuntu Terminal Setup - Astronomi Sınıflandırma Projesi"
echo "=============================================================="

# Project directory check
PROJECT_DIR="$HOME/astro_classification_project"

echo "📁 Proje dizini kontrol ediliyor..."
if [ ! -d "$PROJECT_DIR" ]; then
    echo "❌ Proje dizini bulunamadı: $PROJECT_DIR"
    echo "💡 Projeyi önce Ubuntu'ya kopyalayın:"
    echo "   scp -r /path/to/astro_classification_project user@ubuntu-server:~/"
    exit 1
fi

cd "$PROJECT_DIR"
echo "✅ Proje dizini: $(pwd)"

# Python version check
echo ""
echo "🐍 Python versiyonu kontrol ediliyor..."
PYTHON_VERSION=$(python3 --version 2>&1)
echo "Python versiyonu: $PYTHON_VERSION"

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 bulunamadı. Kurulum yapılıyor..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
fi

# Virtual environment setup
echo ""
echo "🔧 Virtual environment hazırlanıyor..."
if [ ! -d ".venv" ]; then
    echo "Virtual environment oluşturuluyor..."
    python3 -m venv .venv
fi

echo "Virtual environment aktifleştiriliyor..."
source .venv/bin/activate

# Dependencies installation
echo ""
echo "📦 Dependencies kuruluyor..."
pip install --upgrade pip
pip install -r requirements.txt

# Additional Ubuntu-specific packages
echo ""
echo "🔧 Ubuntu-specific paketler kuruluyor..."
sudo apt install -y python3-tk  # matplotlib için
pip install pillow  # PIL için

# GPU support check (optional)
echo ""
echo "🎮 GPU desteği kontrol ediliyor..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU bulundu:"
    nvidia-smi --query-gpu=name --format=csv,noheader
    echo "CUDA desteği için tensorflow-gpu gerekebilir."
else
    echo "ℹ️ GPU bulunamadı, CPU ile çalışacak."
fi

echo ""
echo "✅ Ubuntu setup tamamlandı!"
echo ""
echo "🚀 Kullanım örnekleri:"
echo "# Virtual environment'ı aktifleştir:"
echo "source .venv/bin/activate"
echo ""
echo "# Model test et:"
echo "python3 scripts/test_improved_model.py"
echo ""
echo "# Tek resim tahmin et:"
echo "python3 scripts/improved_predict.py data/split_dataset/test/Galaxy/100263_4e77db06.jpg"
echo ""
echo "# Model yeniden eğit:"
echo "python3 scripts/train_improved_model.py"
