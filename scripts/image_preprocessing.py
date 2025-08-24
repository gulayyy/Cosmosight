import cv2
import os
import uuid

raw_folder = '../data/raw_images'
processed_folder = '../data/processed_images'
target_size = (224, 224)

for category in os.listdir(raw_folder):
    category_path = os.path.join(raw_folder, category)
    if os.path.isdir(category_path):
        save_category_path = os.path.join(processed_folder, category)
        os.makedirs(save_category_path, exist_ok=True)

        count = 0
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)
            if not os.path.isfile(file_path):
                continue

            image = cv2.imread(file_path)
            if image is None:
                print(f"❌ Could not read image: {file_path}")
                continue

            # Renk formatını düzelt (BGR -> RGB)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Resize + blur
            image_resized = cv2.resize(image, target_size)
            image_blur = cv2.GaussianBlur(image_resized, (5, 5), 0)

            # Uzantı kontrolü
            name, ext = os.path.splitext(file_name)
            if ext.lower() not in ['.jpg', '.jpeg', '.png']:
                ext = '.jpg'

            # Benzersiz isim (çakışmayı engellemek için)
            new_filename = f"{name}_{uuid.uuid4().hex[:8]}{ext}"
            save_path = os.path.join(save_category_path, new_filename)

            # Yüksek kaliteyle kaydet
            cv2.imwrite(save_path, cv2.cvtColor(image_blur, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            count += 1

        print(f"✅ {category}: {count} images processed and saved in {save_category_path}")
