"""
ベースラインモデル（Random Forest, SVM, k-NN）の実装
"""
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns


class BaselineModelTrainer:
    """ベースラインモデルの訓練と評価"""

    def __init__(self, model_type: str = 'random_forest',
                 handle_imbalance: bool = True,
                 random_state: int = 42):
        """
        Args:
            model_type: 'random_forest', 'svm', or 'knn'
            handle_imbalance: クラス不均衡を処理するか
            random_state: 乱数シード
        """
        self.model_type = model_type
        self.handle_imbalance = handle_imbalance
        self.random_state = random_state
        self.model = None
        self.class_names = ['No Gesture', 'Fist', 'Wave In',
                           'Wave Out', 'Open', 'Pinch']

    def create_model(self, **kwargs) -> object:
        """モデルを作成"""
        if self.model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', None),
                min_samples_split=kwargs.get('min_samples_split', 2),
                min_samples_leaf=kwargs.get('min_samples_leaf', 1),
                class_weight='balanced' if self.handle_imbalance else None,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )

        elif self.model_type == 'svm':
            model = SVC(
                C=kwargs.get('C', 1.0),
                kernel=kwargs.get('kernel', 'rbf'),
                gamma=kwargs.get('gamma', 'scale'),
                class_weight='balanced' if self.handle_imbalance else None,
                random_state=self.random_state,
                verbose=True
            )

        elif self.model_type == 'knn':
            model = KNeighborsClassifier(
                n_neighbors=kwargs.get('n_neighbors', 5),
                weights=kwargs.get('weights', 'distance'),
                metric=kwargs.get('metric', 'minkowski'),
                n_jobs=-1
            )

        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             **model_kwargs) -> object:
        """
        モデルを訓練

        Args:
            X_train: 訓練データ shape (num_samples, num_features)
            y_train: ラベル shape (num_samples,)
            **model_kwargs: モデル固有のパラメータ

        Returns:
            model: 訓練されたモデル
        """
        print(f"\n{'='*60}")
        print(f"Training {self.model_type.upper()} model")
        print(f"{'='*60}")
        print(f"Training samples: {len(X_train)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Class distribution:")
        for i in range(6):
            count = np.sum(y_train == i)
            print(f"  {self.class_names[i]}: {count} ({count/len(y_train)*100:.1f}%)")

        # モデルを作成
        self.model = self.create_model(**model_kwargs)

        # 訓練
        print(f"\nTraining...")
        self.model.fit(X_train, y_train)
        print("Training completed!")

        return self.model

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        モデルを評価

        Args:
            X_test: テストデータ
            y_test: テストラベル

        Returns:
            metrics: 評価指標の辞書
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_type.upper()} model")
        print(f"{'='*60}")

        # 予測
        y_pred = self.model.predict(X_test)

        # 評価指標を計算
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted', zero_division=0
        )

        # クラスごとの評価
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)

        # 混同行列
        cm = confusion_matrix(y_test, y_pred)

        # 結果を表示
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")

        print(f"\nPer-Class Metrics:")
        for i in range(len(precision_per_class)):
            class_name = self.class_names[i] if i < len(self.class_names) else f"Class_{i}"
            print(f"  {class_name}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall:    {recall_per_class[i]:.4f}")
            print(f"    F1 Score:  {f1_per_class[i]:.4f}")
            print(f"    Support:   {support[i]}")

        # 詳細なクラシフィケーションレポート
        print(f"\nClassification Report:")
        # 実際に存在するクラスのラベルのみを使用
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        label_names = [self.class_names[i] for i in unique_labels if i < len(self.class_names)]
        print(classification_report(y_test, y_pred,
                                   labels=unique_labels,
                                   target_names=label_names,
                                   zero_division=0))

        # メトリクスを辞書にまとめる
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'support': support,
            'confusion_matrix': cm,
            'predictions': y_pred
        }

        return metrics

    def plot_confusion_matrix(self, cm: np.ndarray,
                             save_path: Optional[Path] = None):
        """
        混同行列を可視化

        Args:
            cm: 混同行列
            save_path: 保存先パス
        """
        plt.figure(figsize=(10, 8))

        # パーセンテージと実数を表示
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Percentage'})

        plt.title(f'Confusion Matrix - {self.model_type.upper()}',
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_performance_comparison(self, metrics: Dict,
                                   save_path: Optional[Path] = None):
        """
        クラスごとの性能を比較

        Args:
            metrics: 評価指標の辞書
            save_path: 保存先パス
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics_names = ['Precision', 'Recall', 'F1 Score']
        metrics_data = [
            metrics['precision_per_class'],
            metrics['recall_per_class'],
            metrics['f1_per_class']
        ]

        for idx, (ax, name, data) in enumerate(zip(axes, metrics_names, metrics_data)):
            x = np.arange(len(self.class_names))
            bars = ax.bar(x, data, color='steelblue', alpha=0.7)

            # 値をバーの上に表示
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)

            ax.set_xlabel('Gesture Class', fontsize=12)
            ax.set_ylabel(name, fontsize=12)
            ax.set_title(f'{name} by Class', fontsize=13, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(self.class_names, rotation=45, ha='right')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Performance Metrics - {self.model_type.upper()}',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance comparison saved to: {save_path}")

        plt.show()

    def save_model(self, save_path: Path):
        """モデルを保存"""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'wb') as f:
            pickle.dump(self.model, f)

        print(f"Model saved to: {save_path}")

    def load_model(self, load_path: Path):
        """モデルを読み込む"""
        with open(load_path, 'rb') as f:
            self.model = pickle.load(f)

        print(f"Model loaded from: {load_path}")


if __name__ == "__main__":
    # 使用例
    from sklearn.datasets import make_classification

    # ダミーデータを作成
    X_train, y_train = make_classification(
        n_samples=1000, n_features=120, n_informative=80,
        n_classes=6, n_clusters_per_class=1, class_sep=1.0,
        random_state=42
    )
    X_test, y_test = make_classification(
        n_samples=200, n_features=120, n_informative=80,
        n_classes=6, n_clusters_per_class=1, class_sep=1.0,
        random_state=43
    )

    # Random Forestを訓練
    trainer = BaselineModelTrainer(model_type='random_forest',
                                  handle_imbalance=True)
    trainer.train(X_train, y_train, n_estimators=50)

    # 評価
    metrics = trainer.evaluate(X_test, y_test)

    # 可視化
    trainer.plot_confusion_matrix(metrics['confusion_matrix'])
    trainer.plot_performance_comparison(metrics)
