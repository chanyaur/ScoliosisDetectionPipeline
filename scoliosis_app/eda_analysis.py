"""
Exploratory Data Analysis for Scoliosis1K Dataset
This script analyzes the structure and characteristics of the Scoliosis1K dataset
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")  # makes it nice so there's different colors with all the same properties

class Scoliosis1KEDA:
    def __init__(self, dataset_path='Scoliosis1K-pkl'):
        """
        Initialize EDA class for Scoliosis1K dataset
        
        Args:
            dataset_path: Path to the Scoliosis1K-pkl directory
        """
        self.dataset_path = Path(dataset_path)
        self.data_info = []
        self.class_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
        
    def scan_dataset(self):
        """Scan the entire dataset and collect information"""
        print("Scanning dataset structure...")
        
        # Iterate through all directories
        for subject_dir in tqdm(sorted(self.dataset_path.iterdir())):
            if not subject_dir.is_dir() or subject_dir.name.startswith('.'):
                continue
                
            subject_id = subject_dir.name
            
            # Check for class subdirectories
            for class_dir in subject_dir.iterdir():
                if not class_dir.is_dir() or class_dir.name.startswith('.'):
                    continue
                    
                class_label = class_dir.name
                
                # Count sequences for each class
                for seq_dir in class_dir.iterdir():
                    if not seq_dir.is_dir():
                        continue
                        
                    # Find pickle files
                    pkl_files = list(seq_dir.glob('*.pkl'))
                    
                    for pkl_file in pkl_files:
                        self.data_info.append({
                            'subject_id': subject_id,
                            'class': class_label,
                            'sequence': seq_dir.name,
                            'file_path': str(pkl_file),
                            'file_size': pkl_file.stat().st_size / 1024  # KB
                        })
                        self.class_counts[class_label] += 1
        
        print(f"Found {len(self.data_info)} sequences total")
        return pd.DataFrame(self.data_info)
    
    def analyze_sample_data(self, n_samples=5):
        """Analyze sample pickle files from each class"""
        print("\nAnalyzing sample data structure...")
        
        sample_analysis = {}
        
        for class_label in ['positive', 'neutral', 'negative']:
            # Get samples from this class
            class_samples = [d for d in self.data_info if d['class'] == class_label]
            
            if not class_samples:
                print(f"Warning: No samples found for class '{class_label}'")
                continue
                
            # Take first n samples
            samples_to_analyze = class_samples[:min(n_samples, len(class_samples))]
            
            class_stats = []
            for sample in samples_to_analyze:
                with open(sample['file_path'], 'rb') as f:
                    data = pickle.load(f)
                
                stats = {
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'min': int(np.min(data)),
                    'max': int(np.max(data)),
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'non_zero_ratio': float(np.count_nonzero(data) / data.size)
                }
                class_stats.append(stats)
            
            sample_analysis[class_label] = class_stats
            
            # Print summary for this class
            print(f"\n{class_label.upper()} class samples:")
            print(f"  Shape: {class_stats[0]['shape']}")
            print(f"  Data type: {class_stats[0]['dtype']}")
            print(f"  Value range: [{class_stats[0]['min']}, {class_stats[0]['max']}]")
            print(f"  Mean intensity: {np.mean([s['mean'] for s in class_stats]):.2f}")
            print(f"  Mean std: {np.mean([s['std'] for s in class_stats]):.2f}")
            print(f"  Mean non-zero ratio: {np.mean([s['non_zero_ratio'] for s in class_stats]):.3f}")
        
        return sample_analysis
    
    def visualize_class_distribution(self):
        """Create visualizations for class distribution"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Pie chart
        ax = axes[0]
        colors = ['#ff9999', '#66b3ff', '#99ff99']
        wedges, texts, autotexts = ax.pie(
            self.class_counts.values(),
            labels=self.class_counts.keys(),
            colors=colors,
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title('Class Distribution (Pie Chart)')
        
        # Bar chart
        ax = axes[1]
        bars = ax.bar(self.class_counts.keys(), self.class_counts.values(), color=colors)
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Sequences')
        ax.set_title('Class Distribution (Bar Chart)')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')
        
        # Class imbalance ratio
        ax = axes[2]
        total = sum(self.class_counts.values())
        ratios = {k: v/total for k, v in self.class_counts.items()}
        
        # Create stacked bar for imbalance visualization
        bottom = 0
        for i, (label, ratio) in enumerate(ratios.items()):
            ax.bar(0, ratio, bottom=bottom, color=colors[i], label=label, width=0.5)
            ax.text(0, bottom + ratio/2, f'{label}\n{ratio:.2%}', 
                   ha='center', va='center', fontweight='bold')
            bottom += ratio
        
        ax.set_xlim(-1, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel('Proportion')
        ax.set_title('Class Imbalance Visualization')
        
        plt.tight_layout()
        plt.savefig('scoliosis_app/results/class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def visualize_sample_silhouettes(self, n_samples=3):
        """Visualize sample silhouettes from each class"""
        fig, axes = plt.subplots(3, n_samples * 3, figsize=(15, 10))  # 3 because you want beginning, middle, end
        
        for class_idx, class_label in enumerate(['positive', 'neutral', 'negative']):
            # Get samples from this class
            class_samples = [d for d in self.data_info if d['class'] == class_label]
            
            if not class_samples:  # this is a check so that an empty list isn't returned for a "fake" class
                continue
            
            # Take first n samples
            samples_to_viz = class_samples[:min(n_samples, len(class_samples))]  # takes the first n (3) samples in one class
            
            for sample_idx, sample in enumerate(samples_to_viz):
                with open(sample['file_path'], 'rb') as f:
                    data = pickle.load(f)
                
                # Show frames at beginning, middle, and end
                frame_indices = [0, len(data)//2, len(data)-1]
                
                for frame_idx, frame_num in enumerate(frame_indices):
                    ax_idx = sample_idx * 3 + frame_idx  # to line it up. Ex: for sample 0, first frame --> 0,0
                    ax = axes[class_idx, ax_idx]
                    
                    # Display silhouette
                    ax.imshow(data[frame_num], cmap='gray', vmin=0, vmax=255)
                    ax.axis('off')  # turn off axis so you won't see that it's a graph
                    
                    # Add title
                    if class_idx == 0:
                        time_label = ['Start', 'Middle', 'End'][frame_idx]
                        ax.set_title(f'{time_label} (Frame {frame_num})', fontsize=10)
                    
                    # Add class label on the left
                    if ax_idx == 0:
                        ax.text(-0.1, 0.5, class_label.upper(), 
                               
                            transform=ax.transAxes, fontsize=12,
                               fontweight='bold', va='center', ha='right',
                               rotation=90)
        
        plt.suptitle('Sample Silhouettes Across Classes and Time', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('scoliosis_app/results/sample_silhouettes.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def analyze_temporal_patterns(self, n_samples=10):
        """Analyze temporal patterns in the sequences"""
        print("\nAnalyzing temporal patterns...")
        
        temporal_stats = {}
        
        for class_label in ['positive', 'neutral', 'negative']:
            class_samples = [d for d in self.data_info if d['class'] == class_label][:n_samples]
            
            if not class_samples:
                continue
            
            class_temporal = []
            
            for sample in class_samples:
                with open(sample['file_path'], 'rb') as f:
                    data = pickle.load(f)
                
                # Calculate frame-wise statistics
                frame_means = [np.mean(frame) for frame in data]  # on average (percent -> decimal), how many of the frame numbers are '1'?
                frame_stds = [np.std(frame) for frame in data]  # how much do they differ from the mean...? how is this helpful <--
                
                # Calculate motion (difference between consecutive frames)
                motion = []
                for i in range(1, len(data)):
                    diff = np.abs(data[i].astype(float) - data[i-1].astype(float))
                    motion.append(np.mean(diff))  # the number of pixels that are different
                
                class_temporal.append({
                    'frame_means': frame_means,
                    'frame_stds': frame_stds,
                    'motion': motion
                })
            
            temporal_stats[class_label] = class_temporal
        
        # Visualize temporal patterns
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        for ax_idx, metric in enumerate(['frame_means', 'frame_stds', 'motion']):
            ax = axes[ax_idx]
            
            for class_label, color in zip(['positive', 'neutral', 'negative'], 
                                         ['red', 'blue', 'green']):
                if class_label not in temporal_stats:
                    continue
                    
                # Get all sequences for this class
                if metric == 'motion':
                    sequences = [s['motion'] for s in temporal_stats[class_label]]
                    x_vals = range(1, 300)  # Motion has 299 values
                else:
                    sequences = [s[metric] for s in temporal_stats[class_label]]
                    x_vals = range(300)
                
                # Calculate mean and std across sequences
                mean_seq = np.mean(sequences, axis=0)
                std_seq = np.std(sequences, axis=0)
                
                # Plot mean line
                ax.plot(x_vals, mean_seq, label=class_label, color=color, alpha=0.8)
                
                # Plot confidence interval
                ax.fill_between(x_vals, 
                               mean_seq - std_seq, 
                               mean_seq + std_seq,
                               alpha=0.2, color=color)
            
            ax.set_xlabel('Frame Number')
            ax.set_ylabel(['Mean Intensity', 'Std Intensity', 'Motion (Frame Diff)'][ax_idx])
            ax.set_title(['Temporal Mean Intensity', 'Temporal Std Intensity', 'Temporal Motion'][ax_idx])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Temporal Patterns Across Classes', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('scoliosis_app/results/temporal_patterns.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return temporal_stats
    
    def visualize_area_distribution(self):
        # Example: Analyze silhouette area over time
        # Load a sample
        with open('Scoliosis1K-pkl/00000/positive/000_180/000_180.pkl', 'rb') as f:
            data = pickle.load(f)

        # Calculate silhouette area per frame
        areas = [np.sum(frame > 0) for frame in data]

        print(f'FRAME WITH MIN AREA: {np.argmin(areas)}')
        
        # Plot of Areas Over Time
        plt.figure(figsize=(10, 4))
        plt.plot(areas)
        plt.xlabel('Frame')
        plt.ylabel('Silhouette Area (pixels)')
        plt.title('Silhouette Area Over Time')
        plt.savefig('scoliosis_app/results/area_visualization.png')
        plt.show()
        
        # Plot of Minimum Area Frame
        figs_to_show = [0, 10, 299]
        
        for frame in figs_to_show:
            fig, ax = plt.subplots() 
            ax.imshow(data[frame], cmap='gray', vmin=0, vmax=255)
            ax.axis('off')  # turn off axis so you won't see that it's a graph
            plt.show()
    
    
    
    
    
    
    
    def generate_summary_report(self, df):
        """Generate a summary report of the EDA"""
        report = []
        report.append("=" * 60)
        report.append("SCOLIOSIS1K DATASET - EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Dataset Overview
        report.append("1. DATASET OVERVIEW")
        report.append("-" * 40)
        report.append(f"Total sequences: {len(df)}")
        report.append(f"Total subjects: {df['subject_id'].nunique()}")
        report.append("")
        
        # Class Distribution
        report.append("2. CLASS DISTRIBUTION")
        report.append("-" * 40)
        for class_label, count in self.class_counts.items():
            percentage = (count / len(df)) * 100
            report.append(f"{class_label.capitalize():10s}: {count:4d} sequences ({percentage:.1f}%)")
        
        # Calculate imbalance ratio
        max_class = max(self.class_counts.values())
        min_class = min(self.class_counts.values())
        imbalance_ratio = max_class / min_class
        report.append(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")
        report.append("")
        
        # File Statistics
        report.append("3. FILE STATISTICS")
        report.append("-" * 40)
        report.append(f"Average file size: {df['file_size'].mean():.2f} KB")
        report.append(f"Min file size: {df['file_size'].min():.2f} KB")
        report.append(f"Max file size: {df['file_size'].max():.2f} KB")
        report.append(f"Total dataset size: {df['file_size'].sum()/1024:.2f} MB")
        report.append("")
        
        # Subjects per Class
        report.append("4. SUBJECTS PER CLASS")
        report.append("-" * 40)
        for class_label in ['positive', 'neutral', 'negative']:
            class_df = df[df['class'] == class_label]
            n_subjects = class_df['subject_id'].nunique()
            report.append(f"{class_label.capitalize():10s}: {n_subjects} subjects")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        
        # Print to console
        print("\n" + report_text)
        
        # Save to file
        with open('scoliosis_app/results/eda_report.txt', 'w') as f:
            f.write(report_text)
        
        return report_text

   



def main():
    """Main function to run the complete EDA"""
    # Create results directory
    os.makedirs('scoliosis_app/results', exist_ok=True)
    
    # Initialize EDA
    eda = Scoliosis1KEDA('Scoliosis1K-pkl')
    
    # Scan dataset
    df = eda.scan_dataset()
    
    # Save dataset info to CSV
    df.to_csv('scoliosis_app/results/dataset_info.csv', index=False)
    print(f"Dataset info saved to scoliosis_app/results/dataset_info.csv")
    
    # Analyze sample data
    sample_analysis = eda.analyze_sample_data(n_samples=5)
    
    # # Visualize class distribution
    # print("\nCreating class distribution visualizations...")
    # eda.visualize_class_distribution()
    
    # # Visualize sample silhouettes
    # print("Creating silhouette visualizations...")
    # eda.visualize_sample_silhouettes(n_samples=3)
    
    # # Analyze temporal patterns
    # temporal_stats = eda.analyze_temporal_patterns(n_samples=10)
    
    # Visualize area distribution
    print("Creating area visualizations...")
    eda.visualize_area_distribution()
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = eda.generate_summary_report(df)
    
    print("\n✅ EDA completed successfully!")
    print("Results saved in scoliosis_app/results/")
    
    return df, sample_analysis, temporal_stats


if __name__ == "__main__":
    df, sample_analysis, temporal_stats = main()
