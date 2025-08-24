#!/usr/bin/env python3
"""
🎯 FOR PRESENTATION: Feature Extraction Comparison Module
Analyzing inter-class feature patterns and preparing for presentations
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob

class FeatureComparator:
    """Inter-class feature comparison"""
    
    def __init__(self):
        self.analysis_data = []
        self.class_patterns = {}
    
    def load_analysis_results(self):
        """Load all analysis results"""
        json_files = glob("models/analysis_*.json")
        
        print(f"📊 {len(json_files)} analysis files found")
        
        for file_path in json_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.analysis_data.append(data)
        
        return len(self.analysis_data)
    
    def calculate_class_patterns(self):
        """Calculate class-based patterns"""
        classes = {}
        
        for data in self.analysis_data:
            class_name = data['model_prediction']
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append(data)
        
        # Calculate average values for each class
        for class_name, class_data in classes.items():
            patterns = {
                'count': len(class_data),
                'avg_star_density': np.mean([d['star_density'] for d in class_data]),
                'avg_brightness': np.mean([d['brightness_mean'] for d in class_data]),
                'avg_circularity': np.mean([d['average_circularity'] for d in class_data]),
                'avg_texture_energy': np.mean([d['texture_energy'] for d in class_data]),
                'avg_detected_objects': np.mean([d['detected_objects'] for d in class_data]),
                'avg_contrast': np.mean([d['contrast_ratio'] for d in class_data]),
                'common_shape_category': max(set([d['shape_category'] for d in class_data]), 
                                           key=[d['shape_category'] for d in class_data].count),
                'common_texture_category': max(set([d['texture_category'] for d in class_data]), 
                                             key=[d['texture_category'] for d in class_data].count)
            }
            self.class_patterns[class_name] = patterns
    
    def create_comparison_table(self):
        """Create comparison table for presentations"""
        df_data = []
        
        for class_name, patterns in self.class_patterns.items():
            df_data.append({
                'Class': class_name,
                'Sample Count': patterns['count'],
                'Star Density': f"{patterns['avg_star_density']:.3f}",
                'Average Brightness': f"{patterns['avg_brightness']:.1f}",
                'Symmetry Ratio': f"{patterns['avg_circularity']:.3f}",
                'Texture Complexity': f"{patterns['avg_texture_energy']:.1f}",
                'Detected Objects': f"{patterns['avg_detected_objects']:.1f}",
                'Contrast Ratio': f"{patterns['avg_contrast']:.0f}",
                'Dominant Shape': patterns['common_shape_category'],
                'Dominant Texture': patterns['common_texture_category']
            })
        
        df = pd.DataFrame(df_data)
        
        # Save as CSV
        df.to_csv('models/class_comparison_table.csv', index=False, encoding='utf-8')
        
        print("\n🏆 CLASS COMPARISON TABLE")
        print("="*100)
        print(df.to_string(index=False))
        print("="*100)
        
        return df
    
    def create_pattern_visualization(self):
        """Visualize patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('🌌 Astronomical Object Class Patterns', fontsize=16, fontweight='bold')
        
        classes = list(self.class_patterns.keys())
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        # 1. Star Density Comparison
        star_densities = [self.class_patterns[c]['avg_star_density'] for c in classes]
        bars1 = axes[0, 0].bar(classes, star_densities, color=colors[:len(classes)])
        axes[0, 0].set_title('Star Density by Class')
        axes[0, 0].set_ylabel('Average Star Density')
        for i, v in enumerate(star_densities):
            axes[0, 0].text(i, v + 0.001, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 2. Brightness Comparison
        brightness = [self.class_patterns[c]['avg_brightness'] for c in classes]
        bars2 = axes[0, 1].bar(classes, brightness, color=colors[:len(classes)])
        axes[0, 1].set_title('Brightness by Class')
        axes[0, 1].set_ylabel('Average Brightness (0-255)')
        for i, v in enumerate(brightness):
            axes[0, 1].text(i, v + 2, f'{v:.1f}', ha='center', fontweight='bold')
        
        # 3. Symmetry Comparison
        circularity = [self.class_patterns[c]['avg_circularity'] for c in classes]
        bars3 = axes[1, 0].bar(classes, circularity, color=colors[:len(classes)])
        axes[1, 0].set_title('Circularity (Symmetry) by Class')
        axes[1, 0].set_ylabel('Average Circularity')
        for i, v in enumerate(circularity):
            axes[1, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # 4. Texture Complexity
        texture = [self.class_patterns[c]['avg_texture_energy'] for c in classes]
        bars4 = axes[1, 1].bar(classes, texture, color=colors[:len(classes)])
        axes[1, 1].set_title('Texture Complexity by Class')
        axes[1, 1].set_ylabel('Average Texture Energy')
        for i, v in enumerate(texture):
            axes[1, 1].text(i, v + 5, f'{v:.1f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        save_path = 'models/class_patterns_comparison.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Pattern comparison saved: {save_path}")
        plt.show()
        
        return save_path
    
    def generate_presentation_insights(self):
        """Generate important insights for presentations"""
        insights = []
        
        # Star density analysis
        star_densities = {k: v['avg_star_density'] for k, v in self.class_patterns.items()}
        min_star_class = min(star_densities, key=star_densities.get)
        max_star_class = max(star_densities, key=star_densities.get)
        
        insights.append(f"⭐ Star Density: Highest in {max_star_class} class ({star_densities[max_star_class]:.3f}), lowest in {min_star_class} class ({star_densities[min_star_class]:.3f})")
        
        # Brightness analysis
        brightness = {k: v['avg_brightness'] for k, v in self.class_patterns.items()}
        brightest_class = max(brightness, key=brightness.get)
        darkest_class = min(brightness, key=brightness.get)
        
        insights.append(f"💡 Brightness: {brightest_class} brightest ({brightness[brightest_class]:.1f}/255), {darkest_class} darkest ({brightness[darkest_class]:.1f}/255)")
        
        # Symmetry analysis
        circularity = {k: v['avg_circularity'] for k, v in self.class_patterns.items()}
        most_circular = max(circularity, key=circularity.get)
        least_circular = min(circularity, key=circularity.get)
        
        insights.append(f"🔄 Symmetry: {most_circular} most symmetric ({circularity[most_circular]:.3f}), {least_circular} most irregular ({circularity[least_circular]:.3f})")
        
        # Texture complexity
        texture = {k: v['avg_texture_energy'] for k, v in self.class_patterns.items()}
        most_complex = max(texture, key=texture.get)
        least_complex = min(texture, key=texture.get)
        
        insights.append(f"🎨 Texture: {most_complex} most complex ({texture[most_complex]:.1f}), {least_complex} simplest ({texture[least_complex]:.1f})")
        
        print("\n🧠 IMPORTANT FINDINGS FOR PRESENTATIONS:")
        print("="*80)
        for insight in insights:
            print(f"  {insight}")
        
        # Pattern consistency
        print(f"\n🔬 SCIENTIFIC CONSISTENCY:")
        for class_name, patterns in self.class_patterns.items():
            print(f"  {class_name}: {patterns['common_shape_category']} shape + {patterns['common_texture_category']} texture")
        
        print("="*80)
        
        return insights
    
    def create_presentation_summary(self):
        """Create summary file for presentations"""
        summary = {
            'analysis_date': '2024-08-24',
            'total_images_analyzed': len(self.analysis_data),
            'classes_found': list(self.class_patterns.keys()),
            'class_patterns': self.class_patterns,
            'key_insights': self.generate_presentation_insights()
        }
        
        with open('models/presentation_summary.json', 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Presentation summary saved: models/presentation_summary.json")
        return summary

def main():
    """Main function"""
    comparator = FeatureComparator()
    
    print("🎯 FEATURE EXTRACTION COMPARISON ANALYSIS")
    print("="*60)
    
    # 1. Load analysis files
    loaded_count = comparator.load_analysis_results()
    if loaded_count == 0:
        print("❌ Run presentation_analyzer.py first!")
        return
    
    # 2. Calculate patterns
    comparator.calculate_class_patterns()
    
    # 3. Create comparison table
    df = comparator.create_comparison_table()
    
    # 4. Visualization
    viz_path = comparator.create_pattern_visualization()
    
    # 5. Presentation insights
    insights = comparator.generate_presentation_insights()
    
    # 6. Summary report
    summary = comparator.create_presentation_summary()
    
    print(f"\n🎯 COMPARISON ANALYSIS COMPLETED!")
    print(f"📁 All files in 'models/' folder")
    print(f"📊 Table: class_comparison_table.csv")
    print(f"📈 Graph: class_patterns_comparison.png")
    print(f"📋 Summary: presentation_summary.json")

if __name__ == "__main__":
    main()
