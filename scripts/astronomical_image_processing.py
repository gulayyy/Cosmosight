"""
Astronomical Image Processing and Feature Extraction Module
Advanced algorithms for NASA/ESA quality image analysis
"""

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
# from scipy import ndimage
# from skimage import measure, filters, segmentation

class AstronomyImageProcessor:
    """Specialized processing class for astronomical images"""
    
    def __init__(self):
        self.features_df = pd.DataFrame()
    
    def load_image(self, image_path):
        """Load image and convert to grayscale"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray
    
    def apply_gaussian_filter(self, image, sigma=1.0):
        """Gaussian filtering - noise reduction"""
        # Using OpenCV for Gaussian blur
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image.astype(np.uint8), (kernel_size, kernel_size), sigma)
    
    def detect_edges_canny(self, image, low_threshold=50, high_threshold=150):
        """Canny edge detection - celestial object boundaries"""
        return cv2.Canny(image.astype(np.uint8), low_threshold, high_threshold)
    
    def segment_objects(self, image, method='otsu'):
        """Object segmentation - star/galaxy separation"""
        if method == 'otsu':
            _, binary = cv2.threshold(image.astype(np.uint8), 0, 255, 
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(image.astype(np.uint8), 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        return binary
    
    def extract_star_density(self, image):
        """Star density analysis"""
        # Count bright pixels (stars)
        bright_pixels = np.sum(image > np.percentile(image, 90))
        total_pixels = image.size
        density = bright_pixels / total_pixels
        
        return {
            'star_density': density,
            'bright_pixel_count': bright_pixels,
            'brightness_mean': np.mean(image),
            'brightness_std': np.std(image)
        }
    
    def extract_shape_features(self, binary_image):
        """Extract shape features"""
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        features = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 50:  # Filter small noise
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # Convex hull
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                # Aspect ratio
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Circularity
                circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                features.append({
                    'object_id': i,
                    'area': area,
                    'perimeter': perimeter,
                    'solidity': solidity,
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity,
                    'centroid_x': x + w/2,
                    'centroid_y': y + h/2
                })
        
        return features
    
    def extract_texture_features(self, image):
        """Texture features - for galaxy structures"""
        # GLCM (Gray-Level Co-occurrence Matrix) approach
        # Simplified version
        
        # Gradient magnitude
        grad_x = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Local Binary Pattern like
        texture_energy = np.var(gradient_magnitude)
        texture_contrast = np.max(gradient_magnitude) - np.min(gradient_magnitude)
        
        return {
            'texture_energy': texture_energy,
            'texture_contrast': texture_contrast,
            'gradient_mean': np.mean(gradient_magnitude),
            'gradient_std': np.std(gradient_magnitude)
        }
    
    def process_astronomical_image(self, image_path, save_results=True):
        """Complete astronomical image analysis pipeline"""
        print(f"🔍 Analyzing: {image_path}")
        
        # 1. Load image
        original, gray = self.load_image(image_path)
        
        # 2. Filtering
        filtered = self.apply_gaussian_filter(gray, sigma=1.0)
        
        # 3. Edge detection
        edges = self.detect_edges_canny(filtered)
        
        # 4. Segmentation
        binary = self.segment_objects(filtered)
        
        # 5. Feature extraction
        star_features = self.extract_star_density(gray)
        shape_features = self.extract_shape_features(binary)
        texture_features = self.extract_texture_features(gray)
        
        # 6. Combine results
        results = {
            'image_path': image_path,
            **star_features,
            **texture_features,
            'num_objects': len(shape_features),
            'total_object_area': sum([obj['area'] for obj in shape_features]) if shape_features else 0
        }
        
        # 7. Visualization
        if save_results:
            self.visualize_analysis(original, gray, filtered, edges, binary, image_path)
        
        return results
    
    def visualize_analysis(self, original, gray, filtered, edges, binary, image_path):
        """Visualize analysis results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('Grayscale')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(filtered, cmap='gray')
        axes[0, 2].set_title('Gaussian Filtered')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Canny Edges')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title('Segmented Objects')
        axes[1, 1].axis('off')
        
        # Histogram
        axes[1, 2].hist(gray.ravel(), bins=50, alpha=0.7)
        axes[1, 2].set_title('Pixel Intensity Distribution')
        axes[1, 2].set_xlabel('Intensity')
        axes[1, 2].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        # Save
        import os
        filename = os.path.basename(image_path).split('.')[0]
        plt.savefig(f'models/analysis_{filename}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def batch_process_directory(self, directory_path, output_csv='astronomical_features.csv'):
        """Process all images in directory"""
        import os
        import glob
        
        all_features = []
        
        # Supported formats
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.fits']
        
        for ext in extensions:
            for image_path in glob.glob(os.path.join(directory_path, ext)):
                try:
                    features = self.process_astronomical_image(image_path, save_results=False)
                    all_features.append(features)
                    print(f"✅ Processed: {os.path.basename(image_path)}")
                except Exception as e:
                    print(f"❌ Error processing {image_path}: {e}")
        
        # Convert to DataFrame and save
        if all_features:
            df = pd.DataFrame(all_features)
            df.to_csv(output_csv, index=False)
            print(f"💾 Features saved to: {output_csv}")
            return df
        else:
            print("❌ No features extracted!")
            return None

def main():
    """Example usage"""
    processor = AstronomyImageProcessor()
    
    # Test single image
    test_image = "data/split_dataset/test/Galaxy/100263_4e77db06.jpg"
    
    try:
        features = processor.process_astronomical_image(test_image)
        print("\n📊 Extracted Features:")
        for key, value in features.items():
            print(f"  {key}: {value}")
        
        # Batch processing example
        # df = processor.batch_process_directory("data/split_dataset/test/Galaxy")
        # if df is not None:
        #     print(f"\n📈 Processed {len(df)} images")
        #     print(df.describe())
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
