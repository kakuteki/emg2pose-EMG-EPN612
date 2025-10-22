"""
ベースラインモデル訓練のメインスクリプト
"""
import sys
from pathlib import Path
import numpy as np
import argparse

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(str(Path(__file__).parent / 'src'))

from data.data_loader import EMGDataLoader, create_data_split
from features.feature_extractor import EMGFeatureExtractor, EMGPreprocessor
from models.baseline_models import BaselineModelTrainer


def main(args):
    """メイン処理"""
    print("="*80)
    print("EMG Gesture Recognition - Baseline Model Training")
    print("="*80)

    # ===============================
    # 1. データの読み込み
    # ===============================
    print("\n[Step 1/5] Loading dataset...")
    data_path = args.data_path

    # トレーニングデータを読み込む
    train_loader = EMGDataLoader(data_path, dataset_type='training')
    X_train_raw, y_train, train_user_info = train_loader.load_dataset(
        max_users=args.max_users
    )

    # テストデータを読み込む
    test_loader = EMGDataLoader(data_path, dataset_type='testing')
    X_test_raw, y_test, test_user_info = test_loader.load_dataset(
        max_users=args.max_users
    )

    print(f"\nTraining set: {X_train_raw.shape}")
    print(f"Test set: {X_test_raw.shape}")

    # ===============================
    # 2. 前処理
    # ===============================
    print("\n[Step 2/5] Preprocessing signals...")
    preprocessor = EMGPreprocessor(sampling_rate=200)

    X_train_preprocessed = preprocessor.preprocess(
        X_train_raw,
        apply_bandpass=args.apply_bandpass,
        apply_notch=args.apply_notch,
        normalize=args.normalize
    )

    X_test_preprocessed = preprocessor.preprocess(
        X_test_raw,
        apply_bandpass=args.apply_bandpass,
        apply_notch=args.apply_notch,
        normalize=args.normalize
    )

    print("Preprocessing completed!")

    # ===============================
    # 3. 特徴量抽出
    # ===============================
    print("\n[Step 3/5] Extracting features...")
    feature_extractor = EMGFeatureExtractor(sampling_rate=200)

    print("Extracting training features...")
    X_train_features = feature_extractor.extract_features_batch(
        X_train_preprocessed,
        feature_type=args.feature_type
    )

    print("Extracting test features...")
    X_test_features = feature_extractor.extract_features_batch(
        X_test_preprocessed,
        feature_type=args.feature_type
    )

    print(f"\nFeature shape: {X_train_features.shape}")
    print(f"Number of features per sample: {X_train_features.shape[1]}")

    # 訓練/検証分割
    X_train, X_val, y_train_split, y_val = create_data_split(
        X_train_features, y_train,
        test_size=args.val_split,
        random_state=args.random_state
    )

    print(f"\nFinal split:")
    print(f"  Training: {X_train.shape}")
    print(f"  Validation: {X_val.shape}")
    print(f"  Test: {X_test_features.shape}")

    # ===============================
    # 4. モデルの訓練
    # ===============================
    print("\n[Step 4/5] Training model...")

    trainer = BaselineModelTrainer(
        model_type=args.model_type,
        handle_imbalance=args.handle_imbalance,
        random_state=args.random_state
    )

    # モデル固有のパラメータ
    model_params = {}
    if args.model_type == 'random_forest':
        model_params = {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth
        }
    elif args.model_type == 'svm':
        model_params = {
            'C': args.C,
            'kernel': args.kernel
        }
    elif args.model_type == 'knn':
        model_params = {
            'n_neighbors': args.n_neighbors
        }

    # 訓練
    model = trainer.train(X_train, y_train_split, **model_params)

    # ===============================
    # 5. 評価
    # ===============================
    print("\n[Step 5/5] Evaluating model...")

    # 検証セットで評価
    print("\n--- Validation Set ---")
    val_metrics = trainer.evaluate(X_val, y_val)

    # テストセットで評価
    print("\n--- Test Set ---")
    test_metrics = trainer.evaluate(X_test_features, y_test)

    # ===============================
    # 6. 結果の保存と可視化
    # ===============================
    if args.save_results:
        results_dir = Path('results') / args.model_type
        results_dir.mkdir(parents=True, exist_ok=True)

        # 混同行列を保存
        print("\nSaving confusion matrix...")
        trainer.plot_confusion_matrix(
            test_metrics['confusion_matrix'],
            save_path=results_dir / 'confusion_matrix.png'
        )

        # 性能比較を保存
        print("Saving performance comparison...")
        trainer.plot_performance_comparison(
            test_metrics,
            save_path=results_dir / 'performance_comparison.png'
        )

        # モデルを保存
        if args.save_model:
            model_path = results_dir / f'{args.model_type}_model.pkl'
            trainer.save_model(model_path)

        # メトリクスをテキストファイルに保存
        metrics_path = results_dir / 'metrics.txt'
        with open(metrics_path, 'w') as f:
            f.write(f"Model: {args.model_type}\n")
            f.write(f"{'='*60}\n\n")
            f.write("Test Set Metrics:\n")
            f.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
            f.write(f"  F1 Score:  {test_metrics['f1_score']:.4f}\n")

        print(f"\nResults saved to: {results_dir}")

    print("\n" + "="*80)
    print("Training and evaluation completed!")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train baseline models for EMG gesture recognition'
    )

    # データ関連
    parser.add_argument('--data_path', type=str,
                       default='../EMG-EPN612 Dataset',
                       help='Path to EMG dataset')
    parser.add_argument('--max_users', type=int, default=None,
                       help='Maximum number of users to load (None = all)')

    # 前処理関連
    parser.add_argument('--apply_bandpass', action='store_true', default=True,
                       help='Apply bandpass filter')
    parser.add_argument('--apply_notch', action='store_true', default=True,
                       help='Apply notch filter')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize signals')

    # 特徴量関連
    parser.add_argument('--feature_type', type=str, default='all',
                       choices=['time', 'frequency', 'all'],
                       help='Type of features to extract')

    # モデル関連
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'knn'],
                       help='Type of baseline model')
    parser.add_argument('--handle_imbalance', action='store_true', default=True,
                       help='Handle class imbalance')

    # Random Forest パラメータ
    parser.add_argument('--n_estimators', type=int, default=100,
                       help='Number of trees (Random Forest)')
    parser.add_argument('--max_depth', type=int, default=None,
                       help='Max depth of trees (Random Forest)')

    # SVM パラメータ
    parser.add_argument('--C', type=float, default=1.0,
                       help='Regularization parameter (SVM)')
    parser.add_argument('--kernel', type=str, default='rbf',
                       choices=['linear', 'poly', 'rbf', 'sigmoid'],
                       help='Kernel type (SVM)')

    # k-NN パラメータ
    parser.add_argument('--n_neighbors', type=int, default=5,
                       help='Number of neighbors (k-NN)')

    # その他
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility')
    parser.add_argument('--save_results', action='store_true', default=True,
                       help='Save results and visualizations')
    parser.add_argument('--save_model', action='store_true', default=True,
                       help='Save trained model')

    args = parser.parse_args()

    main(args)
