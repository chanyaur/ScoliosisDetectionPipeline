"""
Deep Analysis of Scoliosis1K Dataset
This script performs comprehensive statistical and temporal analysis
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_sample_data(base_path='Scoliosis1K-pkl', n_samples_per_class=10):
    """Load sample data from each class for analysis"""
    data = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
    
    # Sample indices for each class
    positive_samples = np.random.choice(range(493), n_samples_per_class, replace=False)  # samples cannot be selected w replacement
    neutral_samples = np.random.choice(range(1293, 1493), n_samples_per_class, replace=False)
    negative_samples = np.random.choice(range(493, 1293), n_samples_per_class, replace=False)
    
    # Load positive samples
    for idx in positive_samples:
        path = Path(base_path) / f"{idx:05d}" / "positive" / "000_180" / "000_180.pkl"  # pad the part w 0's and the total length should be 5. d = decimal integer
        if path.exists():
            with open(path, 'rb') as f:
                data['positive'].append(pickle.load(f))
    
    # Load neutral samples
    for idx in neutral_samples:
        path = Path(base_path) / f"{idx:05d}" / "neutral" / "000_180" / "000_180.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                data['neutral'].append(pickle.load(f))
    
    # Load negative samples
    for idx in negative_samples:
        path = Path(base_path) / f"{idx:05d}" / "negative" / "000_180" / "000_180.pkl"
        if path.exists():
            with open(path, 'rb') as f:
                data['negative'].append(pickle.load(f))
    
    return data  # returns a dictionary composed of lists

def extract_gait_features(silhouette_sequence):
    """Extract comprehensive gait features from a silhouette sequence"""
    features = {}
    
    # Convert to numpy array if needed
    seq = np.array(silhouette_sequence)
    T, H, W = seq.shape  # (300, 64, 64)
    
    # print(f"Shape: {seq.shape}")
    
    # 1. Temporal Features
    # Frame-to-frame differences (motion energy)
    frame_diffs = np.diff(seq, axis=0)  # returns an array of differences
    features['mean_motion_energy'] = np.mean(np.abs(frame_diffs))
    features['std_motion_energy'] = np.std(np.abs(frame_diffs))
    features['max_motion_energy'] = np.max(np.abs(frame_diffs))
    
    # 2. Spatial Features
    # Center of mass trajectory
    com_x = []  # com = center of mass
    com_y = []
    for frame in seq:
        if np.sum(frame) > 0:
            y_coords, x_coords = np.where(frame > 0)
            com_x.append(np.mean(x_coords))
            com_y.append(np.mean(y_coords))
        else:
            com_x.append(W/2)
            com_y.append(H/2)
    
    com_x = np.array(com_x)
    com_y = np.array(com_y)
    
    features['com_x_std'] = np.std(com_x)  # how much com_x (vertical center of mass) varies per class
    features['com_y_std'] = np.std(com_y)
    features['com_x_range'] = np.max(com_x) - np.min(com_x)
    features['com_y_range'] = np.max(com_y) - np.min(com_y)
    
    # 3. Shape Features
    # Bounding box dimensions
    widths = []
    heights = []
    aspect_ratios = []
    
    for frame in seq:
        if np.sum(frame) > 0:
            y_coords, x_coords = np.where(frame > 0)
            width = np.max(x_coords) - np.min(x_coords)  # width (of the person)
            height = np.max(y_coords) - np.min(y_coords)
            widths.append(width)
            heights.append(height)
            if height > 0:
                aspect_ratios.append(width / height)
    
    features['mean_width'] = np.mean(widths) if widths else 0
    features['std_width'] = np.std(widths) if widths else 0
    features['mean_height'] = np.mean(heights) if heights else 0
    features['std_height'] = np.std(heights) if heights else 0
    features['mean_aspect_ratio'] = np.mean(aspect_ratios) if aspect_ratios else 0
    features['std_aspect_ratio'] = np.std(aspect_ratios) if aspect_ratios else 0
    
    # 4. Symmetry Features
    # Left-right symmetry
    symmetry_scores = []
    for frame in seq:
        if np.sum(frame) > 0:
            left_half = frame[:, :W//2]
            right_half = frame[:, W//2:]
            right_flipped = np.fliplr(right_half)
            
            # Resize if necessary
            min_width = min(left_half.shape[1], right_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_flipped = right_flipped[:, :min_width]
            
            if np.sum(left_half) > 0 and np.sum(right_flipped) > 0:
                symmetry = np.corrcoef(left_half.flatten(), right_flipped.flatten())[0, 1]  # flip it and see how much overlap they have
                if not np.isnan(symmetry):
                    symmetry_scores.append(symmetry)
    
    features['mean_symmetry'] = np.mean(symmetry_scores) if symmetry_scores else 0
    features['std_symmetry'] = np.std(symmetry_scores) if symmetry_scores else 0
    
    # 5. Periodicity Features (Gait Cycle)
    # Analyze periodicity in silhouette area
    areas = [np.sum(frame > 0) for frame in seq]
    if len(areas) > 20:
        # Find peaks in area signal (corresponding to double support phase)
        peaks, properties = find_peaks(areas, distance=10)  # peaks happen when the area is at local maximum (sum of the numbers is greatest bec it's represented by 1 and 0)
        if len(peaks) > 1:
            step_lengths = np.diff(peaks)  # calculates pixel diff **between consecutive frames**
            features['mean_step_length'] = np.mean(step_lengths)
            features['std_step_length'] = np.std(step_lengths)
            features['gait_regularity'] = 1.0 / (np.std(step_lengths) + 1e-6)  # 1e-6: avoid divide by 0 issues  -->  higher value is more regular
        else:
            features['mean_step_length'] = 0
            features['std_step_length'] = 0
            features['gait_regularity'] = 0
    else:
        features['mean_step_length'] = 0
        features['std_step_length'] = 0
        features['gait_regularity'] = 0
    
    # 6. Upper vs Lower Body Features
    upper_motion = []
    lower_motion = []
    
    for i in range(1, len(seq)):
        diff = np.abs(seq[i] - seq[i-1])  # analyzes the diff within each frame
        upper_diff = np.sum(diff[:H//2, :])
        lower_diff = np.sum(diff[H//2:, :])
        upper_motion.append(upper_diff)
        lower_motion.append(lower_diff)
    
    features['upper_body_motion'] = np.mean(upper_motion) if upper_motion else 0
    features['lower_body_motion'] = np.mean(lower_motion) if lower_motion else 0
    features['upper_lower_ratio'] = (features['upper_body_motion'] / 
                                    (features['lower_body_motion'] + 1e-6))
    
    return features

def perform_statistical_tests(data):
    """Perform statistical tests between classes"""
    results = []
    
    # Extract features for all samples
    features_by_class = {
        'positive': [],
        'neutral': [],
        'negative': []
    }
    
    for class_name in ['positive', 'neutral', 'negative']:
        for sample in data[class_name]:
            features = extract_gait_features(sample)
            features_by_class[class_name].append(features)
    
    # Convert to DataFrames
    dfs = {}
    for class_name in features_by_class:
        if features_by_class[class_name]:
            dfs[class_name] = pd.DataFrame(features_by_class[class_name])
    
    # Perform ANOVA for each feature
    # F-statistics (between group and within group variance - ex between pos/neg vs within pos class)
    # p-value (probability diffs occurred by chance)
    # eta squared: what proportion of variances can be explained?
    
    
    feature_names = list(features.keys())
    
    for feature in feature_names:
        try:
            positive_vals = dfs['positive'][feature].values
            neutral_vals = dfs['neutral'][feature].values
            negative_vals = dfs['negative'][feature].values
            
            # ANOVA test
            f_stat, p_val = stats.f_oneway(positive_vals, neutral_vals, negative_vals)
            
            # Effect size (eta squared)
            grand_mean = np.mean(np.concatenate([positive_vals, neutral_vals, negative_vals]))  # average value of all values over all classes?
            ss_between = (len(positive_vals) * (np.mean(positive_vals) - grand_mean)**2 +  # how much each group's mean deviates from the grand mean
                         len(neutral_vals) * (np.mean(neutral_vals) - grand_mean)**2 +  # aka the between group sum of squares
                         len(negative_vals) * (np.mean(negative_vals) - grand_mean)**2)
            # sum of squares of all groups
            ss_total = np.sum((positive_vals - grand_mean)**2) + \
                      np.sum((neutral_vals - grand_mean)**2) + \
                      np.sum((negative_vals - grand_mean)**2)
            eta_squared = ss_between / (ss_total + 1e-6)
            # A higher η² value indicates that the independent variable accounts for a greater percentage of the total variability in the dependent variable
            
            
            results.append({
                'feature': feature,
                'f_statistic': f_stat,
                'p_value': p_val,
                'eta_squared': eta_squared,
                'significant': p_val < 0.05,
                'mean_positive': np.mean(positive_vals),
                'mean_neutral': np.mean(neutral_vals),
                'mean_negative': np.mean(negative_vals),
                'std_positive': np.std(positive_vals),
                'std_neutral': np.std(neutral_vals),
                'std_negative': np.std(negative_vals)
            })
        except Exception as e:
            print(f"Error processing feature {feature}: {e}")
    
    return pd.DataFrame(results), dfs

def create_deep_visualizations(data, results_df, feature_dfs):
    """Create comprehensive visualizations"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Feature Importance (based on eta squared)
    ax1 = plt.subplot(3, 3, 1)
    significant_features = results_df[results_df['significant']].sort_values('eta_squared', ascending=False)
    if not significant_features.empty:
        top_features = significant_features.head(10)
        ax1.barh(range(len(top_features)), top_features['eta_squared'].values)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features['feature'].values, fontsize=8)
        ax1.set_xlabel('Effect Size (η²)')
        ax1.set_title('Top Discriminative Features')
    
    # 2. P-value distribution
    ax2 = plt.subplot(3, 3, 2)
    ax2.hist(results_df['p_value'], bins=20, edgecolor='black')
    ax2.axvline(x=0.05, color='r', linestyle='--', label='α=0.05')
    ax2.set_xlabel('P-value')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Statistical Significance Distribution')
    ax2.legend()
    
    # 3. Symmetry Analysis
    ax3 = plt.subplot(3, 3, 3)
    symmetry_data = []
    labels = []
    for class_name in ['positive', 'neutral', 'negative']:
        if class_name in feature_dfs and 'mean_symmetry' in feature_dfs[class_name]:
            symmetry_data.append(feature_dfs[class_name]['mean_symmetry'].values)
            labels.append(class_name)
    
    if symmetry_data:
        bp = ax3.boxplot(symmetry_data, labels=labels)
        ax3.set_ylabel('Symmetry Score')
        ax3.set_title('Left-Right Symmetry by Class')
        ax3.grid(True, alpha=0.3)
    
    # 4. Motion Energy Distribution
    ax4 = plt.subplot(3, 3, 4)
    for class_name in ['positive', 'neutral', 'negative']:
        if class_name in feature_dfs and 'mean_motion_energy' in feature_dfs[class_name]:
            ax4.hist(feature_dfs[class_name]['mean_motion_energy'], 
                    alpha=0.5, label=class_name, bins=15)
    ax4.set_xlabel('Mean Motion Energy')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Motion Energy Distribution')
    ax4.legend()
    
    # 5. Gait Regularity
    ax5 = plt.subplot(3, 3, 5)
    regularity_data = []
    labels = []
    for class_name in ['positive', 'neutral', 'negative']:
        if class_name in feature_dfs and 'gait_regularity' in feature_dfs[class_name]:
            regularity_data.append(feature_dfs[class_name]['gait_regularity'].values)
            labels.append(class_name)
    
    if regularity_data:
        vp = ax5.violinplot(regularity_data, positions=range(len(labels)), 
                           showmeans=True, showmedians=True)
        ax5.set_xticks(range(len(labels)))
        ax5.set_xticklabels(labels)
        ax5.set_ylabel('Gait Regularity')
        ax5.set_title('Gait Cycle Regularity')
    
    # 6. Upper vs Lower Body Motion Ratio
    ax6 = plt.subplot(3, 3, 6)
    for class_name in ['positive', 'neutral', 'negative']:
        if class_name in feature_dfs and 'upper_lower_ratio' in feature_dfs[class_name]:
            ax6.scatter(feature_dfs[class_name]['upper_body_motion'],
                       feature_dfs[class_name]['lower_body_motion'],
                       label=class_name, alpha=0.6, s=50)
    ax6.set_xlabel('Upper Body Motion')
    ax6.set_ylabel('Lower Body Motion')
    ax6.set_title('Upper vs Lower Body Motion')
    ax6.legend()
    
    # 7. Feature Correlation Matrix
    ax7 = plt.subplot(3, 3, 7)
    # Combine all features
    all_features = pd.concat([feature_dfs[c] for c in feature_dfs], ignore_index=True)
    if not all_features.empty:
        corr_matrix = all_features.select_dtypes(include=[np.number]).corr()
        im = ax7.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax7.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)
    
    # 8. Center of Mass Variability
    ax8 = plt.subplot(3, 3, 8)
    com_data = []
    labels = []
    for class_name in ['positive', 'neutral', 'negative']:
        if class_name in feature_dfs and 'com_x_std' in feature_dfs[class_name]:
            com_combined = np.sqrt(feature_dfs[class_name]['com_x_std']**2 + 
                                  feature_dfs[class_name]['com_y_std']**2)
            com_data.append(com_combined.values)
            labels.append(class_name)
    
    if com_data:
        bp = ax8.boxplot(com_data, labels=labels, patch_artist=True)
        colors = ['lightcoral', 'lightgray', 'lightblue']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        ax8.set_ylabel('CoM Variability')
        ax8.set_title('Center of Mass Stability')
    
    # 9. Aspect Ratio Analysis
    ax9 = plt.subplot(3, 3, 9)
    for class_name in ['positive', 'neutral', 'negative']:
        if class_name in feature_dfs and 'mean_aspect_ratio' in feature_dfs[class_name]:
            ax9.hist(feature_dfs[class_name]['mean_aspect_ratio'], 
                    alpha=0.5, label=class_name, bins=15)
    ax9.set_xlabel('Mean Aspect Ratio (W/H)')
    ax9.set_ylabel('Frequency')
    ax9.set_title('Body Aspect Ratio Distribution')
    ax9.legend()
    
    plt.suptitle('Deep Analysis: Scoliosis1K Dataset Features', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('scoliosis_app/results/deep_analysis_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def perform_dimensionality_reduction(feature_dfs):
    """Perform PCA and t-SNE for visualization"""
    
    # Prepare data
    X_list = []
    y_list = []
    class_map = {'positive': 0, 'neutral': 1, 'negative': 2}
    
    for class_name in ['positive', 'neutral', 'negative']:
        if class_name in feature_dfs:
            X_list.append(feature_dfs[class_name].values)
            y_list.extend([class_map[class_name]] * len(feature_dfs[class_name]))
    
    if not X_list:
        return None
    
    X = np.vstack(X_list)
    y = np.array(y_list)
    
    # Handle NaN and inf values
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
    X_tsne = tsne.fit_transform(X)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PCA plot
    colors = ['red', 'gray', 'blue']
    labels = ['Positive', 'Neutral', 'Negative']
    
    for i in range(3):
        mask = y == i
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                   c=colors[i], label=labels[i], alpha=0.6, s=100)
    
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax1.set_title('PCA: Feature Space Visualization')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # t-SNE plot
    for i in range(3):
        mask = y == i
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=colors[i], label=labels[i], alpha=0.6, s=100)
    
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.set_title('t-SNE: Non-linear Manifold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Dimensionality Reduction of Gait Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('scoliosis_app/results/dimensionality_reduction.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return X_pca, X_tsne, y

def main():
    print("=" * 60)
    print("DEEP ANALYSIS OF SCOLIOSIS1K DATASET")
    print("=" * 60)
    
    # Load sample data
    print("\n1. Loading sample data...")
    data = load_sample_data(n_samples_per_class=20)
    print(f"   Loaded: {len(data['positive'])} positive, "
          f"{len(data['neutral'])} neutral, {len(data['negative'])} negative samples")
    
    # Perform statistical analysis
    print("\n2. Extracting features and performing statistical tests...")
    results_df, feature_dfs = perform_statistical_tests(data)
    
    # Save statistical results
    results_df.to_csv('scoliosis_app/results/statistical_analysis.csv', index=False)
    print("   Statistical results saved to statistical_analysis.csv")
    
    # Print significant features  -- when you train the model, it will likely look more closely at these features
    print("\n3. Most Discriminative Features (p < 0.05):")
    significant = results_df[results_df['significant']].sort_values('eta_squared', ascending=False)
    for _, row in significant.head(10).iterrows():
        print(f"   - {row['feature']}: η²={row['eta_squared']:.4f}, p={row['p_value']:.4e}")
    
    # Create visualizations
    print("\n4. Creating comprehensive visualizations...")
    create_deep_visualizations(data, results_df, feature_dfs)
    
    # Dimensionality reduction
    print("\n5. Performing dimensionality reduction...")
    X_pca, X_tsne, y = perform_dimensionality_reduction(feature_dfs)
    
    # Generate summary statistics
    print("\n6. Summary Statistics:")
    print("-" * 40)
    
    total_features = len(results_df)
    significant_features = len(results_df[results_df['significant']])
    
    print(f"   Total features analyzed: {total_features}")
    print(f"   Significant features (p<0.05): {significant_features} ({100*significant_features/total_features:.1f}%)")
    
    # Top discriminative features
    top_features = significant.head(5)
    print("\n   Top 5 Most Discriminative Features:")
    for _, row in top_features.iterrows():
        print(f"   • {row['feature']}:")
        print(f"     - Positive: {row['mean_positive']:.3f} ± {row['std_positive']:.3f}")
        print(f"     - Neutral:  {row['mean_neutral']:.3f} ± {row['std_neutral']:.3f}")
        print(f"     - Negative: {row['mean_negative']:.3f} ± {row['std_negative']:.3f}")
    
    print("\n" + "=" * 60)
    print("Deep analysis completed successfully!")
    print("Results saved in scoliosis_app/results/")
    print("=" * 60)

if __name__ == "__main__":
    main()
