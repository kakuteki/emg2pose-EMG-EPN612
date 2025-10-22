import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)

class EMGDatasetAnalyzer:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.training_path = self.base_path / "trainingJSON"
        self.testing_path = self.base_path / "testingJSON"
        self.gesture_labels = {
            0: "noGesture",
            1: "fist",
            2: "waveIn",
            3: "waveOut",
            4: "open",
            5: "pinch"
        }

    def load_user_data(self, user_folder):
        """Load data from a single user folder"""
        json_file = user_folder / f"{user_folder.name}.json"

        if not json_file.exists():
            return None

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            return None

    def get_dataset_overview(self):
        """Get overview of the dataset structure"""
        print("=" * 80)
        print("EMG-EPN612 Dataset Overview")
        print("=" * 80)

        train_users = sorted([d for d in self.training_path.iterdir() if d.is_dir()])
        test_users = sorted([d for d in self.testing_path.iterdir() if d.is_dir()])

        print(f"\nTraining users: {len(train_users)}")
        print(f"Testing users: {len(test_users)}")
        print(f"Total users: {len(train_users) + len(test_users)}")

        return train_users, test_users

    def analyze_user_demographics(self, users, dataset_name):
        """Analyze user demographic information"""
        print(f"\n{'-' * 80}")
        print(f"{dataset_name} User Demographics")
        print(f"{'-' * 80}")

        demographics = []

        for user_folder in users[:100]:  # Limit to first 100 users for speed
            data = self.load_user_data(user_folder)
            if data and 'userInfo' in data:
                demographics.append(data['userInfo'])

        df = pd.DataFrame(demographics)

        # Age statistics
        if 'age' in df.columns:
            print(f"\nAge Statistics:")
            print(f"  Mean: {df['age'].mean():.1f}")
            print(f"  Median: {df['age'].median():.1f}")
            print(f"  Range: {df['age'].min()} - {df['age'].max()}")

        # Gender distribution
        if 'gender' in df.columns:
            print(f"\nGender Distribution:")
            gender_counts = df['gender'].value_counts()
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count} ({count/len(df)*100:.1f}%)")

        # Handedness distribution
        if 'handedness' in df.columns:
            print(f"\nHandedness Distribution:")
            hand_counts = df['handedness'].value_counts()
            for hand, count in hand_counts.items():
                print(f"  {hand}: {count} ({count/len(df)*100:.1f}%)")

        return df

    def analyze_emg_signals(self, users, dataset_name, num_samples=10):
        """Analyze EMG signal characteristics"""
        print(f"\n{'-' * 80}")
        print(f"{dataset_name} EMG Signal Analysis")
        print(f"{'-' * 80}")

        emg_stats = []
        gesture_counts = defaultdict(int)

        for i, user_folder in enumerate(users[:num_samples]):
            data = self.load_user_data(user_folder)
            if not data:
                continue

            # General info
            if i == 0:
                print(f"\nGeneral Information:")
                print(f"  Device: {data['generalInfo']['deviceModel']}")
                print(f"  Sampling Frequency: {data['generalInfo']['samplingFrequencyInHertz']} Hz")
                print(f"  Recording Time: {data['generalInfo']['recordingTimeInSeconds']} seconds")

            # Analyze synchronization gesture
            if 'synchronizationGesture' in data:
                sync_data = data['synchronizationGesture']['samples']

                for sample_key in sync_data:
                    sample = sync_data[sample_key]

                    # Count gestures
                    if 'myoDetection' in sample:
                        for gesture in sample['myoDetection']:
                            gesture_counts[gesture] += 1

                    # EMG statistics
                    if 'emg' in sample:
                        emg = sample['emg']
                        for ch_name in ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']:
                            if ch_name in emg:
                                ch_data = np.array(emg[ch_name])
                                emg_stats.append({
                                    'user': user_folder.name,
                                    'channel': ch_name,
                                    'mean': np.mean(ch_data),
                                    'std': np.std(ch_data),
                                    'min': np.min(ch_data),
                                    'max': np.max(ch_data),
                                    'range': np.max(ch_data) - np.min(ch_data)
                                })

        # EMG signal statistics
        print(f"\nEMG Signal Statistics (based on {num_samples} users):")
        df_emg = pd.DataFrame(emg_stats)

        if not df_emg.empty:
            print(f"\nOverall EMG Statistics:")
            print(f"  Mean amplitude: {df_emg['mean'].mean():.2f} ± {df_emg['mean'].std():.2f}")
            print(f"  Std deviation: {df_emg['std'].mean():.2f} ± {df_emg['std'].std():.2f}")
            print(f"  Signal range: {df_emg['range'].mean():.2f} ± {df_emg['range'].std():.2f}")

            print(f"\nPer-Channel Statistics:")
            channel_stats = df_emg.groupby('channel').agg({
                'mean': ['mean', 'std'],
                'std': ['mean', 'std'],
                'range': ['mean', 'std']
            }).round(2)
            print(channel_stats)

        # Gesture distribution
        print(f"\nGesture Distribution:")
        total_gestures = sum(gesture_counts.values())
        for gesture_id in sorted(gesture_counts.keys()):
            gesture_name = self.gesture_labels.get(gesture_id, f"Unknown_{gesture_id}")
            count = gesture_counts[gesture_id]
            percentage = (count / total_gestures * 100) if total_gestures > 0 else 0
            print(f"  {gesture_name} ({gesture_id}): {count} ({percentage:.1f}%)")

        return df_emg, gesture_counts

    def visualize_sample_emg(self, users, output_path):
        """Visualize sample EMG signals"""
        print(f"\n{'-' * 80}")
        print(f"Creating EMG Signal Visualizations")
        print(f"{'-' * 80}")

        # Select first user with valid data
        sample_data = None
        for user_folder in users[:10]:
            data = self.load_user_data(user_folder)
            if data and 'synchronizationGesture' in data:
                sample_data = data
                print(f"\nUsing data from: {user_folder.name}")
                break

        if not sample_data:
            print("No valid data found for visualization")
            return

        # Get first sample
        sync_data = sample_data['synchronizationGesture']['samples']
        first_sample_key = list(sync_data.keys())[0]
        sample = sync_data[first_sample_key]

        # Plot EMG channels
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        fig.suptitle('EMG Signal - All 8 Channels', fontsize=16, fontweight='bold')

        channels = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

        for idx, ch_name in enumerate(channels):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            if ch_name in sample['emg']:
                signal = np.array(sample['emg'][ch_name])
                time = np.arange(len(signal)) / 200.0  # 200 Hz sampling rate

                ax.plot(time, signal, linewidth=0.5)
                ax.set_title(f'{ch_name.upper()}', fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Amplitude')
                ax.grid(True, alpha=0.3)

                # Add statistics to plot
                mean_val = np.mean(signal)
                std_val = np.std(signal)
                ax.text(0.02, 0.98, f'Mean: {mean_val:.1f}\nStd: {std_val:.1f}',
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_path / 'emg_signals_8channels.png', dpi=300, bbox_inches='tight')
        print(f"Saved: emg_signals_8channels.png")
        plt.close()

        # Plot gesture detection
        if 'myoDetection' in sample:
            fig, ax = plt.subplots(figsize=(15, 4))
            gestures = np.array(sample['myoDetection'])
            time = np.arange(len(gestures)) / 200.0

            # Create color map for gestures
            colors = {0: 'gray', 1: 'red', 2: 'blue', 3: 'green', 4: 'orange', 5: 'purple'}

            for gesture_id in np.unique(gestures):
                mask = gestures == gesture_id
                gesture_name = self.gesture_labels[gesture_id]
                ax.scatter(time[mask], gestures[mask],
                          label=gesture_name,
                          color=colors.get(gesture_id, 'black'),
                          alpha=0.6, s=10)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Gesture ID')
            ax.set_title('Myo Gesture Detection Over Time', fontweight='bold')
            ax.set_ylim(-0.5, 5.5)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / 'gesture_detection_timeline.png', dpi=300, bbox_inches='tight')
            print(f"Saved: gesture_detection_timeline.png")
            plt.close()

    def create_summary_visualizations(self, df_demographics, df_emg, gesture_counts, output_path):
        """Create summary visualizations"""
        print(f"\n{'-' * 80}")
        print(f"Creating Summary Visualizations")
        print(f"{'-' * 80}")

        # Demographics plots
        if not df_demographics.empty:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('User Demographics Summary', fontsize=16, fontweight='bold')

            # Age distribution
            if 'age' in df_demographics.columns:
                axes[0, 0].hist(df_demographics['age'], bins=20, color='skyblue', edgecolor='black')
                axes[0, 0].set_xlabel('Age')
                axes[0, 0].set_ylabel('Frequency')
                axes[0, 0].set_title('Age Distribution')
                axes[0, 0].grid(True, alpha=0.3)

            # Gender distribution
            if 'gender' in df_demographics.columns:
                gender_counts = df_demographics['gender'].value_counts()
                axes[0, 1].bar(gender_counts.index, gender_counts.values, color=['lightblue', 'lightpink'])
                axes[0, 1].set_xlabel('Gender')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Gender Distribution')
                axes[0, 1].grid(True, alpha=0.3, axis='y')

            # Handedness distribution
            if 'handedness' in df_demographics.columns:
                hand_counts = df_demographics['handedness'].value_counts()
                axes[1, 0].bar(hand_counts.index, hand_counts.values, color='lightgreen')
                axes[1, 0].set_xlabel('Handedness')
                axes[1, 0].set_ylabel('Count')
                axes[1, 0].set_title('Handedness Distribution')
                axes[1, 0].grid(True, alpha=0.3, axis='y')

            # Ethnic group distribution
            if 'ethnicGroup' in df_demographics.columns:
                ethnic_counts = df_demographics['ethnicGroup'].value_counts()
                axes[1, 1].bar(range(len(ethnic_counts)), ethnic_counts.values, color='lightyellow')
                axes[1, 1].set_xticks(range(len(ethnic_counts)))
                axes[1, 1].set_xticklabels(ethnic_counts.index, rotation=45, ha='right')
                axes[1, 1].set_xlabel('Ethnic Group')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Ethnic Group Distribution')
                axes[1, 1].grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(output_path / 'demographics_summary.png', dpi=300, bbox_inches='tight')
            print(f"Saved: demographics_summary.png")
            plt.close()

        # EMG statistics plots
        if not df_emg.empty:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('EMG Signal Statistics', fontsize=16, fontweight='bold')

            # Channel-wise mean comparison
            channel_means = df_emg.groupby('channel')['mean'].mean().sort_index()
            axes[0, 0].bar(range(len(channel_means)), channel_means.values, color='steelblue')
            axes[0, 0].set_xticks(range(len(channel_means)))
            axes[0, 0].set_xticklabels(channel_means.index, rotation=45)
            axes[0, 0].set_xlabel('Channel')
            axes[0, 0].set_ylabel('Mean Amplitude')
            axes[0, 0].set_title('Mean EMG Amplitude by Channel')
            axes[0, 0].grid(True, alpha=0.3, axis='y')

            # Channel-wise std comparison
            channel_stds = df_emg.groupby('channel')['std'].mean().sort_index()
            axes[0, 1].bar(range(len(channel_stds)), channel_stds.values, color='coral')
            axes[0, 1].set_xticks(range(len(channel_stds)))
            axes[0, 1].set_xticklabels(channel_stds.index, rotation=45)
            axes[0, 1].set_xlabel('Channel')
            axes[0, 1].set_ylabel('Standard Deviation')
            axes[0, 1].set_title('EMG Signal Variability by Channel')
            axes[0, 1].grid(True, alpha=0.3, axis='y')

            # Signal range comparison
            channel_ranges = df_emg.groupby('channel')['range'].mean().sort_index()
            axes[1, 0].bar(range(len(channel_ranges)), channel_ranges.values, color='lightgreen')
            axes[1, 0].set_xticks(range(len(channel_ranges)))
            axes[1, 0].set_xticklabels(channel_ranges.index, rotation=45)
            axes[1, 0].set_xlabel('Channel')
            axes[1, 0].set_ylabel('Signal Range')
            axes[1, 0].set_title('EMG Signal Range by Channel')
            axes[1, 0].grid(True, alpha=0.3, axis='y')

            # Overall amplitude distribution
            axes[1, 1].hist(df_emg['mean'], bins=30, color='plum', edgecolor='black')
            axes[1, 1].set_xlabel('Mean Amplitude')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of EMG Mean Amplitudes')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path / 'emg_statistics_summary.png', dpi=300, bbox_inches='tight')
            print(f"Saved: emg_statistics_summary.png")
            plt.close()

        # Gesture distribution plot
        if gesture_counts:
            fig, ax = plt.subplots(figsize=(10, 6))

            gesture_names = [self.gesture_labels.get(gid, f"Unknown_{gid}") for gid in sorted(gesture_counts.keys())]
            gesture_values = [gesture_counts[gid] for gid in sorted(gesture_counts.keys())]

            colors_list = ['gray', 'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
            colors_to_use = colors_list[:len(gesture_names)] if len(gesture_names) <= len(colors_list) else plt.cm.tab10(range(len(gesture_names)))
            bars = ax.bar(gesture_names, gesture_values, color=colors_to_use)

            ax.set_xlabel('Gesture Type', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Gesture Distribution in Dataset', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(output_path / 'gesture_distribution.png', dpi=300, bbox_inches='tight')
            print(f"Saved: gesture_distribution.png")
            plt.close()

    def run_full_analysis(self):
        """Run complete EDA analysis"""
        print("\n" + "=" * 80)
        print("Starting Comprehensive EMG Dataset Analysis")
        print("=" * 80)

        # Create output directory
        output_path = self.base_path / "eda_results"
        output_path.mkdir(exist_ok=True)

        # Get dataset overview
        train_users, test_users = self.get_dataset_overview()

        # Analyze demographics
        df_demographics_train = self.analyze_user_demographics(train_users, "Training")

        # Analyze EMG signals
        df_emg_train, gesture_counts_train = self.analyze_emg_signals(train_users, "Training", num_samples=20)

        # Create visualizations
        self.visualize_sample_emg(train_users, output_path)
        self.create_summary_visualizations(df_demographics_train, df_emg_train, gesture_counts_train, output_path)

        # Save statistical summaries
        if not df_demographics_train.empty:
            df_demographics_train.describe().to_csv(output_path / 'demographics_stats.csv')
            print(f"\nSaved: demographics_stats.csv")

        if not df_emg_train.empty:
            df_emg_train.describe().to_csv(output_path / 'emg_stats.csv')
            print(f"Saved: emg_stats.csv")

        print("\n" + "=" * 80)
        print(f"Analysis Complete! Results saved to: {output_path}")
        print("=" * 80)

        return output_path

if __name__ == "__main__":
    # Initialize analyzer
    base_path = Path(__file__).parent
    analyzer = EMGDatasetAnalyzer(base_path)

    # Run full analysis
    output_path = analyzer.run_full_analysis()

    print(f"\n\nAll results have been saved to: {output_path}")
    print("\nGenerated files:")
    print("  - emg_signals_8channels.png")
    print("  - gesture_detection_timeline.png")
    print("  - demographics_summary.png")
    print("  - emg_statistics_summary.png")
    print("  - gesture_distribution.png")
    print("  - demographics_stats.csv")
    print("  - emg_stats.csv")
