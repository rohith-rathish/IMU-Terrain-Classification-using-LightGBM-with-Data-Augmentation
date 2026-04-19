import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import sys

# ============================================================================
# IMU DATA AUGMENTATION SCRIPT (Windows/Linux Compatible)
# Augments 50% of IMU dataset values with 10% scaling + random variation
# ============================================================================

def get_output_directory():
    """
    Automatically detect the output directory based on OS and script location.
    Creates the directory if it doesn't exist.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.resolve()
    
    # Create an 'output' subdirectory in the same location as the script
    output_dir = script_dir / 'augmented_output'
    output_dir.mkdir(exist_ok=True)
    
    return output_dir


def get_input_file():
    """
    Automatically detect the input file location.
    Looks for imu_dataset_original.csv in the script directory or parent directories.
    """
    script_dir = Path(__file__).parent.resolve()
    
    # Check in script directory
    input_file = script_dir / 'imu_dataset_original.csv'
    if input_file.exists():
        return input_file
    
    # Check in parent directory
    input_file = script_dir.parent / 'imu_dataset_original.csv'
    if input_file.exists():
        return input_file
    
    # Check in Desktop
    desktop = Path.home() / 'Desktop' / 'imu_dataset_original.csv'
    if desktop.exists():
        return desktop
    
    # If not found, return default in script directory
    return script_dir / 'imu_dataset_original.csv'


def augment_imu_dataset(dataset, augmentation_percentage=0.5, scaling_factor=0.1, 
                        noise_range=(-0.05, 0.05), random_state=42):
    """
    Augment IMU dataset by scaling and adding noise to a percentage of values.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input IMU dataset to augment
    augmentation_percentage : float
        Percentage of rows to augment (default: 0.5 for 50%)
    scaling_factor : float
        Scaling factor to apply (default: 0.1 for 10%)
    noise_range : tuple
        Range for random noise to add (min, max)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    augmented_df : pd.DataFrame
        Original dataset with augmented rows appended
    """
    
    np.random.seed(random_state)
    
    # Create a copy to avoid modifying original
    original_df = dataset.copy()
    
    # Calculate number of rows to augment
    num_rows = len(original_df)
    num_rows_to_augment = int(num_rows * augmentation_percentage)
    
    # Randomly select which rows to augment
    augment_indices = np.random.choice(num_rows, num_rows_to_augment, replace=False)
    
    # Sensor columns to augment (exclude time, terrain, and label columns)
    sensor_columns = ['wx', 'wy', 'wz', 'ax', 'ay', 'az']
    
    # Create augmented rows
    augmented_rows = []
    
    for idx in augment_indices:
        # Get the row to augment
        row = original_df.iloc[idx].copy()
        
        # For sensor columns only
        for col in sensor_columns:
            # Apply 10% scaling
            scaled_value = row[col] * (1 + scaling_factor)
            
            # Add random noise
            noise = np.random.uniform(noise_range[0], noise_range[1])
            augmented_value = scaled_value + noise
            
            row[col] = augmented_value
        
        augmented_rows.append(row)
    
    # Convert augmented rows to DataFrame
    augmented_df = pd.DataFrame(augmented_rows).reset_index(drop=True)
    
    # Combine original and augmented datasets
    final_augmented_dataset = pd.concat([original_df, augmented_df], ignore_index=True)
    
    print(f"✓ IMU Data Augmentation Complete!")
    print(f"  Original dataset size: {len(original_df)} rows")
    print(f"  Rows augmented: {num_rows_to_augment} rows ({augmentation_percentage*100}%)")
    print(f"  Final augmented dataset size: {len(final_augmented_dataset)} rows")
    
    return final_augmented_dataset


def analyze_augmentation(original_df, augmented_df):
    """
    Analyze and compare original vs augmented data statistics.
    
    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataset
    augmented_df : pd.DataFrame
        Augmented dataset
    """
    
    sensor_columns = ['wx', 'wy', 'wz', 'ax', 'ay', 'az']
    
    print("\n" + "="*70)
    print("DATA AUGMENTATION ANALYSIS")
    print("="*70)
    
    print("\n[ORIGINAL DATA STATISTICS]")
    print(original_df[sensor_columns].describe().round(4))
    
    print("\n[AUGMENTED DATA STATISTICS]")
    print(augmented_df[sensor_columns].describe().round(4))
    
    print("\n[TERRAIN DISTRIBUTION - ORIGINAL]")
    print(original_df['Terrain'].value_counts())
    
    print("\n[TERRAIN DISTRIBUTION - AUGMENTED]")
    print(augmented_df['Terrain'].value_counts())
    
    # Calculate differences in means
    print("\n[MEAN VALUE DIFFERENCES (Augmented - Original)]")
    original_means = original_df[sensor_columns].mean()
    augmented_means = augmented_df[sensor_columns].mean()
    
    for col in sensor_columns:
        diff = augmented_means[col] - original_means[col]
        pct_diff = (diff / abs(original_means[col]) * 100) if original_means[col] != 0 else 0
        print(f"  {col:5s}: {diff:10.6f} ({pct_diff:+6.2f}%)")


def visualize_augmentation(original_df, augmented_df, output_path):
    """
    Create visualization comparing original vs augmented data.
    
    Parameters:
    -----------
    original_df : pd.DataFrame
        Original dataset
    augmented_df : pd.DataFrame
        Augmented dataset
    output_path : Path
        Path to save the visualization
    """
    
    sensor_columns = ['wx', 'wy', 'wz', 'ax', 'ay', 'az']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('IMU Data Augmentation: Original vs Augmented', fontsize=16, fontweight='bold')
    
    for idx, col in enumerate(sensor_columns):
        row = idx // 3
        col_idx = idx % 3
        ax = axes[row, col_idx]
        
        # Plot distributions
        ax.hist(original_df[col], bins=50, alpha=0.6, label='Original', color='blue', edgecolor='black')
        ax.hist(augmented_df[col], bins=50, alpha=0.6, label='Augmented', color='red', edgecolor='black')
        
        ax.set_xlabel(f'{col} (m/s² or rad/s)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{col} Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    print(f"✓ Visualization saved: {output_path}")
    plt.close()


def main():
    """Main execution function."""
    
    # Get paths
    output_dir = get_output_directory()
    input_file = get_input_file()
    
    print("=" * 70)
    print("IMU DATASET AUGMENTATION (Windows/Linux Compatible)")
    print("=" * 70)
    print(f"\n📂 Script location: {Path(__file__).parent.resolve()}")
    print(f"📂 Output directory: {output_dir}")
    print(f"📂 Input file: {input_file}")
    
    # Check if input file exists
    if not input_file.exists():
        print(f"\n❌ ERROR: Input file not found!")
        print(f"   Expected at: {input_file}")
        print(f"\n💡 Please make sure 'imu_dataset_original.csv' is in:")
        print(f"   - Same directory as this script")
        print(f"   - Parent directory of this script")
        print(f"   - Your Desktop")
        sys.exit(1)
    
    print(f"\n✓ Input file found: {input_file}")
    
    # Load IMU dataset
    print(f"\n📂 Loading dataset...")
    
    try:
        original_dataset = pd.read_csv(str(input_file))
    except Exception as e:
        print(f"❌ ERROR reading CSV: {e}")
        sys.exit(1)
    
    print(f"\n📊 Dataset Info:")
    print(f"   Rows: {len(original_dataset)}")
    print(f"   Columns: {list(original_dataset.columns)}")
    print(f"   Memory usage: {original_dataset.memory_usage().sum() / 1024**2:.2f} MB")
    
    print(f"\n🌍 Terrain Types:")
    print(original_dataset['Terrain'].value_counts())
    
    print(f"\n📈 Sample Data (first 5 rows):")
    print(original_dataset.head())
    
    # Perform augmentation
    print(f"\n🔄 Performing augmentation (50% scaling, 10% increase)...")
    
    augmented_dataset = augment_imu_dataset(
        dataset=original_dataset,
        augmentation_percentage=0.5,      # Augment 50% of rows
        scaling_factor=0.1,               # 10% scaling
        noise_range=(-0.05, 0.05),        # ±5% noise
        random_state=42
    )
    
    # Analyze augmentation
    analyze_augmentation(original_dataset, augmented_dataset)
    
    # Save augmented dataset
    output_csv = output_dir / 'augmented_dataset.csv'
    try:
        augmented_dataset.to_csv(str(output_csv), index=False)
        print(f"\n✓ Augmented dataset saved: {output_csv}")
    except Exception as e:
        print(f"❌ ERROR saving CSV: {e}")
        sys.exit(1)
    
    # Save comparison statistics
    comparison_file = output_dir / 'augmentation_statistics.txt'
    try:
        with open(str(comparison_file), 'w') as f:
            f.write("="*70 + "\n")
            f.write("IMU DATA AUGMENTATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Original Dataset Size: {len(original_dataset)} rows\n")
            f.write(f"Augmented Rows Added: {len(augmented_dataset) - len(original_dataset)} rows\n")
            f.write(f"Final Dataset Size: {len(augmented_dataset)} rows\n")
            f.write(f"Augmentation Percentage: 50%\n")
            f.write(f"Scaling Factor: 10%\n")
            f.write(f"Noise Range: ±5%\n\n")
            
            f.write("ORIGINAL DATA STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(str(original_dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].describe().round(4)))
            f.write("\n\n")
            
            f.write("AUGMENTED DATA STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(str(augmented_dataset[['wx', 'wy', 'wz', 'ax', 'ay', 'az']].describe().round(4)))
            f.write("\n\n")
            
            f.write("TERRAIN DISTRIBUTION - ORIGINAL\n")
            f.write("-"*70 + "\n")
            f.write(str(original_dataset['Terrain'].value_counts()))
            f.write("\n\n")
            
            f.write("TERRAIN DISTRIBUTION - AUGMENTED\n")
            f.write("-"*70 + "\n")
            f.write(str(augmented_dataset['Terrain'].value_counts()))
        
        print(f"✓ Statistics saved: {comparison_file}")
    except Exception as e:
        print(f"❌ ERROR saving statistics: {e}")
        sys.exit(1)
    
    # Create visualization
    print(f"\n📊 Creating visualization...")
    try:
        visualization_file = output_dir / 'augmentation_comparison.png'
        visualize_augmentation(
            original_dataset, 
            augmented_dataset,
            visualization_file
        )
    except Exception as e:
        print(f"⚠️  Warning: Could not create visualization: {e}")
    
    # Show sample of augmented data
    print("\n📝 Sample of AUGMENTED DATA (last 5 rows):")
    print(augmented_dataset.tail())
    
    print("\n" + "="*70)
    print("✅ AUGMENTATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output Directory: {output_dir}")
    print("\n📁 Output Files:")
    print("   1. augmented_dataset.csv - Full augmented dataset")
    print("   2. augmentation_statistics.txt - Detailed statistics")
    print("   3. augmentation_comparison.png - Distribution comparison")
    print("\n")


if __name__ == "__main__":
    main()