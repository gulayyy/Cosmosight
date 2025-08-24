#!/usr/bin/env python3
"""
🎯 FOR PRESENTATION: Astronomical Image Analysis & Feature Extraction
Supporting model predictions with scientific analysis module
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import os
import json

class PresentationAnalyzer:
    """Optimized astronomical image analysis for presentations"""
    
    def __init__(self):
        self.results = {}
    
    def analyze_for_presentation(self, image_path, model_prediction=None, confidence=None):
        """Comprehensive analysis for presentations"""
        print(f"🔬 Scientific Analysis: {os.path.basename(image_path)}")
        print("="*50)
        
        # 1. Görüntü yükleme
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Temel özellikler
        basic_features = self._extract_basic_features(gray)
        
        # 3. Yıldız analizi
        star_analysis = self._analyze_stars(gray)
        
        # 4. Şekil analizi
        shape_analysis = self._analyze_shapes(gray)
        
        # 5. Doku analizi
        texture_analysis = self._analyze_texture(gray)
        
        # 6. Sonuçları birleştir
        analysis = {
            'image_path': image_path,
            'model_prediction': model_prediction,
            'confidence': confidence,
            **basic_features,
            **star_analysis,
            **shape_analysis,
            **texture_analysis
        }
        
        # 7. Sunum raporu oluştur
        self._generate_presentation_report(analysis, image_path)
        
        return analysis
    
    def _extract_basic_features(self, gray_image):
        """Basic image properties"""
        return {
            'brightness_mean': float(np.mean(gray_image)),
            'brightness_std': float(np.std(gray_image)),
            'contrast_ratio': float(np.max(gray_image) - np.min(gray_image)),
            'image_quality': 'High' if np.std(gray_image) > 30 else 'Medium'
        }
    
    def _analyze_stars(self, gray_image):
        """Star density and distribution analysis"""
        # Consider bright pixels as stars
        bright_threshold = np.percentile(gray_image, 95)
        bright_pixels = np.sum(gray_image > bright_threshold)
        total_pixels = gray_image.size
        
        # Calculate star density
        star_density = bright_pixels / total_pixels
        
        # Determine star category
        if star_density > 0.15:
            star_category = "Dense Star Field"
        elif star_density > 0.05:
            star_category = "Moderate Density"
        else:
            star_category = "Sparse Star Field"
        
        return {
            'star_density': float(star_density),
            'estimated_star_count': int(bright_pixels / 10),  # Rough estimate
            'star_category': star_category,
            'brightness_distribution': 'Uniform' if np.std(gray_image) < 50 else 'Varied'
        }
    
    def _analyze_shapes(self, gray_image):
        """Shape and object analysis"""
        # Binary threshold
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Contour bulma
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Büyük nesneleri filtrele
        significant_objects = [c for c in contours if cv2.contourArea(c) > 100]
        
        # Shape analysis
        total_area = sum(cv2.contourArea(c) for c in significant_objects)
        avg_circularity = 0
        
        for contour in significant_objects:
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter ** 2)
                avg_circularity += circularity
        
        avg_circularity = avg_circularity / max(len(significant_objects), 1)
        
        # Shape category
        if avg_circularity > 0.7:
            shape_category = "Regular/Spherical"
        elif avg_circularity > 0.4:
            shape_category = "Partially Regular"
        else:
            shape_category = "Irregular/Spiral"
        
        return {
            'detected_objects': len(significant_objects),
            'total_object_area': int(total_area),
            'average_circularity': float(avg_circularity),
            'shape_category': shape_category,
            'complexity': 'High' if len(significant_objects) > 20 else 'Low'
        }
    
    def _analyze_texture(self, gray_image):
        """Texture and structure analysis"""
        # Gradient analizi
        grad_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        texture_energy = np.var(gradient_magnitude)
        edge_density = np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 90)) / gradient_magnitude.size
        
        # Texture category
        if texture_energy > 1000:
            texture_category = "High Detail"
        elif texture_energy > 400:
            texture_category = "Medium Detail"
        else:
            texture_category = "Low Detail"
        
        return {
            'texture_energy': float(texture_energy),
            'edge_density': float(edge_density),
            'texture_category': texture_category,
            'surface_complexity': 'Smooth' if texture_energy < 500 else 'Complex'
        }
    
    def _generate_presentation_report(self, analysis, image_path):
        """Generate formatted report for presentations"""
        filename = os.path.basename(image_path).split('.')[0]
        
        print(f"\n🎯 PRESENTATION REPORT: {filename}")
        print("="*60)
        
        if analysis['model_prediction']:
            print(f"🤖 AI Prediction: {analysis['model_prediction']} ({analysis.get('confidence', 0):.1f}% confidence)")
        
        print(f"\n📊 SCIENTIFIC ANALYSIS:")
        print(f"  🌟 Star Category: {analysis['star_category']}")
        print(f"  🔢 Estimated Star Count: {analysis['estimated_star_count']}")
        print(f"  📐 Shape Category: {analysis['shape_category']}")
        print(f"  🎨 Texture Category: {analysis['texture_category']}")
        print(f"  🔍 Detected Objects: {analysis['detected_objects']} items")
        
        print(f"\n📈 TECHNICAL DETAILS:")
        print(f"  💡 Average Brightness: {analysis['brightness_mean']:.1f}/255")
        print(f"  📊 Contrast Ratio: {analysis['contrast_ratio']:.0f}")
        print(f"  🎯 Star Density: {analysis['star_density']:.3f}")
        print(f"  🔄 Average Symmetry: {analysis['average_circularity']:.3f}")
        
        # Save results as JSON
        output_file = f"models/analysis_{filename}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Detailed analysis saved: {output_file}")
    
    def create_presentation_visualization(self, image_path, analysis):
        """Create visualization for presentations"""
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'🌌 Astronomical Analysis: {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
        
        # Orijinal görüntü
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Original Image\nPrediction: {analysis.get("model_prediction", "Unknown")}')
        axes[0, 0].axis('off')
        
        # Parlaklık analizi
        axes[0, 1].hist(gray.ravel(), bins=50, alpha=0.7, color='skyblue')
        axes[0, 1].axvline(analysis['brightness_mean'], color='red', linestyle='--', 
                          label=f'Mean: {analysis["brightness_mean"]:.1f}')
        axes[0, 1].set_title('Brightness Distribution')
        axes[0, 1].set_xlabel('Pixel Intensity')
        axes[0, 1].legend()
        
        # Yıldız yoğunluğu gösterimi
        bright_threshold = np.percentile(gray, 95)
        bright_mask = gray > bright_threshold
        axes[1, 0].imshow(bright_mask, cmap='Reds')
        axes[1, 0].set_title(f'Star Detection\nCount: {analysis["estimated_star_count"]}')
        axes[1, 0].axis('off')
        
        # Şekil analizi
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        axes[1, 1].imshow(binary, cmap='gray')
        axes[1, 1].set_title(f'Shape Analysis\nObjects: {analysis["detected_objects"]}')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Kaydet
        filename = os.path.basename(image_path).split('.')[0]
        save_path = f"models/presentation_analysis_{filename}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Presentation visual saved: {save_path}")
        plt.show()
        
        return save_path
    
    def demo_for_presentation(self):
        """Demo for presentations"""
        print("🎯 STARTING PRESENTATION DEMO...")
        print("="*60)
        
        # Test görüntüleri
        test_images = [
            ("data/split_dataset/test/Galaxy/100263_4e77db06.jpg", "Galaxy", 98.0),
            ("data/split_dataset/test/Nebula/10002074_large2x_27414cf3.jpeg", "Nebula", 96.0),
            ("data/split_dataset/test/Star/fitscut (10).jpeg", "Star", 89.0)
        ]
        
        results = []
        
        for img_path, prediction, confidence in test_images:
            if os.path.exists(img_path):
                print(f"\n{'='*60}")
                analysis = self.analyze_for_presentation(img_path, prediction, confidence)
                viz_path = self.create_presentation_visualization(img_path, analysis)
                results.append((analysis, viz_path))
                print(f"{'='*60}")
            else:
                print(f"❌ File not found: {img_path}")
        
        # Summary report
        print(f"\n🎯 PRESENTATION SUMMARY:")
        print(f"✅ {len(results)} images analyzed")
        print(f"📊 All analyses in 'models/' folder")
        print(f"🎨 Visualizations ready")
        
        return results

def main():
    """Main function for presentations"""
    analyzer = PresentationAnalyzer()
    
    print("🌌 ASTRONOMICAL IMAGE ANALYSIS - PRESENTATION MODEL")
    print("="*60)
    print("With this module you can in your presentation:")
    print("✅ Support AI predictions with scientific data")
    print("✅ Explain physical properties in images")  
    print("✅ Present details like star density, shape analysis")
    print("="*60)
    
    # Run demo
    results = analyzer.demo_for_presentation()
    
    print(f"\n🎯 READY FOR PRESENTATION!")
    print(f"📁 All files in 'models/' folder")

if __name__ == "__main__":
    main()
