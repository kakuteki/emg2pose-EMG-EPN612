"""
特徴量選択による本質的な特徴抽出

精度を落とさずに、最も重要な特徴量のみを選択する

Trial 43: Top 30 features
Trial 44: Top 50 features
Trial 45: Top 70 features
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid tkinter errors
import matplotlib.pyplot as plt

sys.path.insert(0, 'src')

from data.data_loader import EMGDataLoader, create_data_split
from features.feature_extractor import EMGFeatureExtractor, EMGPreprocessor


def get_feature_names():
    """特徴量の名前リストを生成"""
    feature_names = []

    # 時間領域特徴 (80特徴 = 8チャンネル × 10特徴)
    time_features = ['MAV', 'WL', 'ZC', 'SSC', 'RMS', 'VAR', 'IEMG', 'DASDV', 'LOG', 'WA']
    for ch in range(8):
        for feat in time_features:
            feature_names.append(f'ch{ch}_{feat}')

    # 周波数領域特徴 (40特徴 = 8チャンネル × 5特徴)
    freq_features = ['MNF', 'MDF', 'PKF', 'MNPWR', 'TTP']
    for ch in range(8):
        for feat in freq_features:
            feature_names.append(f'ch{ch}_{feat}')

    return feature_names


def extract_features_from_dataset(X, feature_extractor, preprocessor):
    """データセット全体から手作業特徴量を抽出"""
    print(f"\n特徴量抽出中... (入力サイズ: {X.shape})")

    # 前処理
    X_preprocessed = preprocessor.preprocess(X)

    # バッチで特徴量抽出
    features = feature_extractor.extract_features_batch(
        X_preprocessed,
        feature_type='all'
    )

    print(f"抽出した特徴量のサイズ: {features.shape}")

    return features


def analyze_feature_importance(X_train, y_train, X_val, y_val, feature_names):
    """Random ForestとXGBoostで特徴量重要度を分析"""
    print("\n" + "="*80)
    print("特徴量重要度分析")
    print("="*80)

    # Random Forest
    print("\nRandom Forestで特徴量重要度を計算中...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        random_state=789,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_importance = rf.feature_importances_

    val_pred = rf.predict(X_val)
    rf_acc = accuracy_score(y_val, val_pred) * 100
    print(f"Random Forest Validation Accuracy: {rf_acc:.2f}%")

    # XGBoost
    print("\nXGBoostで特徴量重要度を計算中...")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        random_state=789,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    xgb_importance = xgb.feature_importances_

    val_pred = xgb.predict(X_val)
    xgb_acc = accuracy_score(y_val, val_pred) * 100
    print(f"XGBoost Validation Accuracy: {xgb_acc:.2f}%")

    # 2つのモデルの重要度を平均
    combined_importance = (rf_importance + xgb_importance) / 2

    # 特徴量重要度のDataFrame作成
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': rf_importance,
        'xgb_importance': xgb_importance,
        'combined_importance': combined_importance
    })

    # 重要度でソート
    importance_df = importance_df.sort_values('combined_importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(importance_df.head(20).to_string(index=False))

    return importance_df


def train_with_selected_features(X_train, y_train, X_val, y_val, X_test, y_test,
                                  feature_indices, trial_name, n_features):
    """選択された特徴量だけで訓練"""
    print("\n" + "="*80)
    print(f"{trial_name}: Top {n_features} Features")
    print("="*80)

    # 特徴量を選択
    X_train_selected = X_train[:, feature_indices]
    X_val_selected = X_val[:, feature_indices]
    X_test_selected = X_test[:, feature_indices]

    print(f"\n選択後の特徴量数: {X_train_selected.shape[1]}")

    results = {}

    # Random Forest
    print("\n--- Random Forest ---")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=789,
        n_jobs=-1
    )
    rf.fit(X_train_selected, y_train)

    val_pred = rf.predict(X_val_selected)
    test_pred = rf.predict(X_test_selected)

    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100

    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

    results['random_forest'] = {
        'method': f'Random Forest (Top {n_features})',
        'n_features': n_features,
        'val_acc': val_acc,
        'test_acc': test_acc
    }

    # XGBoost
    print("\n--- XGBoost ---")
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=789,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train_selected, y_train)

    val_pred = xgb.predict(X_val_selected)
    test_pred = xgb.predict(X_test_selected)

    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100

    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

    results['xgboost'] = {
        'method': f'XGBoost (Top {n_features})',
        'n_features': n_features,
        'val_acc': val_acc,
        'test_acc': test_acc
    }

    # Ensemble
    print("\n--- Ensemble ---")
    from sklearn.ensemble import VotingClassifier
    from sklearn.svm import SVC

    svm = SVC(C=10.0, kernel='rbf', gamma='scale', random_state=789)

    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('xgb', xgb), ('svm', svm)],
        voting='hard'
    )
    ensemble.fit(X_train_selected, y_train)

    val_pred = ensemble.predict(X_val_selected)
    test_pred = ensemble.predict(X_test_selected)

    val_acc = accuracy_score(y_val, val_pred) * 100
    test_acc = accuracy_score(y_test, test_pred) * 100

    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")

    results['ensemble'] = {
        'method': f'Ensemble (Top {n_features})',
        'n_features': n_features,
        'val_acc': val_acc,
        'test_acc': test_acc
    }

    return results


def plot_feature_importance(importance_df, save_path='results/feature_importance.png'):
    """特徴量重要度をプロット"""
    plt.figure(figsize=(12, 8))

    top_30 = importance_df.head(30)

    plt.barh(range(len(top_30)), top_30['combined_importance'].values)
    plt.yticks(range(len(top_30)), top_30['feature'].values)
    plt.xlabel('Combined Importance (RF + XGB) / 2')
    plt.ylabel('Feature')
    plt.title('Top 30 Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n特徴量重要度のプロットを保存: {save_path}")
    plt.close()


def main():
    print("="*80)
    print("Feature Selection for Essential Features")
    print("="*80)

    # データロード
    print("\nデータ読み込み中...")
    train_loader_data = EMGDataLoader('.', dataset_type='training')
    test_loader_data = EMGDataLoader('.', dataset_type='testing')

    X_train, y_train, _ = train_loader_data.load_dataset()
    X_test, y_test, _ = test_loader_data.load_dataset()

    # Pinchクラスを除外
    print("\nPinchクラス(label 5)を除外中...")
    train_mask = y_train != 5
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    test_mask = y_test != 5
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]

    print(f"訓練データ: {X_train.shape}")
    print(f"テストデータ: {X_test.shape}")

    # Train/Val分割
    X_train_split, X_val, y_train_split, y_val = create_data_split(
        X_train, y_train, test_size=0.2, random_state=789
    )

    print(f"\n分割後:")
    print(f"  Train: {X_train_split.shape}")
    print(f"  Val: {X_val.shape}")
    print(f"  Test: {X_test.shape}")

    # 特徴量抽出
    feature_extractor = EMGFeatureExtractor(sampling_rate=200)
    preprocessor = EMGPreprocessor(sampling_rate=200)

    print("\n" + "="*80)
    print("手作業特徴量抽出")
    print("="*80)

    X_train_features = extract_features_from_dataset(X_train_split, feature_extractor, preprocessor)
    X_val_features = extract_features_from_dataset(X_val, feature_extractor, preprocessor)
    X_test_features = extract_features_from_dataset(X_test, feature_extractor, preprocessor)

    # NaNやInfを0で置換
    X_train_features = np.nan_to_num(X_train_features, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_features = np.nan_to_num(X_val_features, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_features = np.nan_to_num(X_test_features, nan=0.0, posinf=0.0, neginf=0.0)

    # 特徴量名を取得
    feature_names = get_feature_names()

    # 特徴量重要度を分析
    importance_df = analyze_feature_importance(
        X_train_features, y_train_split,
        X_val_features, y_val,
        feature_names
    )

    # 特徴量重要度をプロット
    plot_feature_importance(importance_df)

    # 特徴量重要度をCSVで保存
    importance_df.to_csv('results/feature_importance.csv', index=False)
    print("\n特徴量重要度を保存: results/feature_importance.csv")

    # 様々な特徴数で実験
    all_results = {}

    for n_features, trial_name in [(30, 'trial43_top30'),
                                     (50, 'trial44_top50'),
                                     (70, 'trial45_top70')]:
        # Top N特徴のインデックスを取得
        top_indices = importance_df.head(n_features).index.tolist()

        # Top N特徴の名前を表示
        top_features = importance_df.head(n_features)['feature'].tolist()
        print(f"\n{trial_name} - Selected Features:")
        for i, feat in enumerate(top_features, 1):
            importance = importance_df[importance_df['feature'] == feat]['combined_importance'].values[0]
            print(f"  {i:2d}. {feat:20s} (importance: {importance:.6f})")

        # 選択した特徴で訓練
        results = train_with_selected_features(
            X_train_features, y_train_split,
            X_val_features, y_val,
            X_test_features, y_test,
            top_indices, trial_name, n_features
        )

        all_results[trial_name] = results

    # 結果サマリー
    print("\n" + "="*80)
    print("Results Summary")
    print("="*80)

    # ベースライン (Trial 39-42の結果)
    baseline_results = {
        'trial39_random_forest_120': {'method': 'Random Forest (120 features)', 'n_features': 120, 'test_acc': 72.08},
        'trial40_xgboost_120': {'method': 'XGBoost (120 features)', 'n_features': 120, 'test_acc': 71.34},
        'trial42_ensemble_120': {'method': 'Ensemble (120 features)', 'n_features': 120, 'test_acc': 72.11}
    }

    print("\nBaseline (All 120 Features):")
    for trial_name, result in baseline_results.items():
        print(f"  {result['method']:35s}: {result['test_acc']:.2f}%")

    print("\nWith Feature Selection:")
    for trial_name, trial_results in all_results.items():
        print(f"\n{trial_name}:")
        for model_name, result in trial_results.items():
            print(f"  {result['method']:35s}: Test {result['test_acc']:.2f}%")

    # JSON形式で保存
    save_results = {}
    for trial_name, trial_results in all_results.items():
        for model_name, result in trial_results.items():
            key = f"{trial_name}_{model_name}"
            save_results[key] = result

    os.makedirs('results', exist_ok=True)
    with open('results/feature_selection_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print("\n結果を保存: results/feature_selection_results.json")

    # 比較表を作成
    print("\n" + "="*80)
    print("Comparison: Feature Selection vs Baseline")
    print("="*80)

    comparison_data = []

    # Baseline
    comparison_data.append({
        'Model': 'Random Forest',
        '120 features': 72.08,
        '70 features': all_results['trial45_top70']['random_forest']['test_acc'],
        '50 features': all_results['trial44_top50']['random_forest']['test_acc'],
        '30 features': all_results['trial43_top30']['random_forest']['test_acc']
    })

    comparison_data.append({
        'Model': 'XGBoost',
        '120 features': 71.34,
        '70 features': all_results['trial45_top70']['xgboost']['test_acc'],
        '50 features': all_results['trial44_top50']['xgboost']['test_acc'],
        '30 features': all_results['trial43_top30']['xgboost']['test_acc']
    })

    comparison_data.append({
        'Model': 'Ensemble',
        '120 features': 72.11,
        '70 features': all_results['trial45_top70']['ensemble']['test_acc'],
        '50 features': all_results['trial44_top50']['ensemble']['test_acc'],
        '30 features': all_results['trial43_top30']['ensemble']['test_acc']
    })

    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + comparison_df.to_string(index=False))

    # WaveFormerとの比較
    print("\n" + "="*80)
    print("Comparison with WaveFormer Baseline (72.33%)")
    print("="*80)

    waveformer_baseline = 72.33

    for trial_name, trial_results in all_results.items():
        n_features = trial_results['random_forest']['n_features']
        print(f"\nTop {n_features} Features:")
        for model_name, result in trial_results.items():
            test_acc = result['test_acc']
            diff = test_acc - waveformer_baseline
            status = "BETTER" if diff > 0 else "SAME" if abs(diff) < 0.01 else "WORSE"
            print(f"  {result['method']:35s}: {test_acc:.2f}% ({diff:+.2f}%) [{status}]")


if __name__ == "__main__":
    main()
