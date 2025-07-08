# AstroVision

## Proje Hakkında
**AstroVision**, astronomik görüntüleri analiz ederek yapay zeka ve görüntü işleme teknolojilerinin gücüyle anlamlı sonuçlar elde eden bir sistemdir. Proje, yıldız kümelerini sınıflandırma, galaksileri tanımlama ve teleskop verilerini işleme üzerine odaklanmıştır.

Bu proje, uzay araştırmalarında kullanılabilecek bir yapay zeka modeli geliştirmeyi ve astronomik verilerin daha anlaşılır hale getirilmesini hedeflemektedir.

---

## Proje Özellikleri
- **Astronomik Veri İşleme**: Açık veri kaynaklarından alınan teleskop verilerinin analizi.
- **Görüntü İşleme**: Kenar algılama, segmentasyon ve filtreleme gibi temel görüntü işleme teknikleri.
- **Yapay Zeka ile Sınıflandırma**: Basit bir CNN modeli kullanarak yıldız kümelerini ve galaksileri sınıflandırma.
- **Sonuçların Görselleştirilmesi**: Model performansı ve analiz sonuçlarının görselleştirilmesi.

---

## Teknolojiler
Proje aşağıdaki teknolojiler ve araçlar kullanılarak geliştirilmiştir:
- **Programlama Dili**: Python
- **Görüntü İşleme**: OpenCV, Pillow
- **Yapay Zeka ve Derin Öğrenme**: TensorFlow, PyTorch, Keras
- **Veri Analizi ve Manipülasyonu**: Pandas, NumPy
- **Görselleştirme**: Matplotlib, Seaborn, Plotly
- **Astronomik Hesaplamalar**: AstroPy, AstroML

---

## Veri Kaynakları
Proje için kullanılabilecek açık astronomik veri kaynakları:
- [NASA Hubble Data Archive](https://archive.stsci.edu/hst/)
- [Sloan Digital Sky Survey (SDSS)](https://www.sdss.org/dr16/)
- [ESA Gaia Archive](https://gea.esac.esa.int/archive/)
- [Kaggle Astronomical Data Sets](https://www.kaggle.com)

---

## Kurulum

### Gereksinimler
Projenin çalıştırılabilmesi için aşağıdaki araçların kurulmuş olması gerekmektedir:
- Python (versiyon 3.8 veya üzeri)
- Pip (Python paket yöneticisi)

### Adımlar
1. Depoyu klonlayın:
    ```bash
    git clone https://github.com/gulayyy/astrovision.git
    ```
2. Proje dizinine gidin:
    ```bash
    cd astrovision
    ```
3. Gerekli Python kütüphanelerini yüklemek için aşağıdaki komutu çalıştırın:
    ```bash
    pip install -r requirements.txt
    ```

---

## Kullanım
Projenin ana işlevlerini çalıştırmak için aşağıdaki adımları takip edin:

1. **Veri Seti Hazırlığı**:
   - Veri setlerini `data/` klasörüne yerleştirin.
   - Veri setlerini uygun formatta düzenleyin.

2. **Görüntü İşleme**:
   - Görüntü filtreleme, kenar algılama ve segmentasyon için `image_processing.py` dosyasını çalıştırın:
     ```bash
     python image_processing.py
     ```

3. **Model Eğitimi**:
   - Yapay zeka modelini eğitmek için aşağıdaki komutu çalıştırın:
     ```bash
     python train_model.py
     ```

4. **Sonuçların Görselleştirilmesi**:
   - Model sonuçlarını görselleştirmek için `visualize_results.py` dosyasını çalıştırın:
     ```bash
     python visualize_results.py
     ```

---

## Katkıda Bulunma
Bu projeye katkıda bulunmak için aşağıdaki adımları izleyebilirsiniz:

1. Depoyu `fork` edin.
2. Yeni bir `branch` oluşturun:
    ```bash
    git checkout -b feature-branch
    ```
3. Değişikliklerinizi yapın ve commit edin:
    ```bash
    git commit -m "Yeni özellik eklendi."
    ```
4. Değişikliklerinizi ana depoya `pull request` olarak gönderin.

---

## Lisans
Bu proje [MIT Lisansı](LICENSE) altında lisanslanmıştır. Detaylı bilgi için `LICENSE` dosyasını inceleyebilirsiniz.

---

## İletişim
Proje hakkında sorularınız için bana ulaşabilirsiniz:
- GitHub: [gulayyy](https://github.com/gulayyy)
- E-posta: gulay@example.com
