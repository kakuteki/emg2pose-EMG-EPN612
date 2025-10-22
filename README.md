# EMG-EPN612 データセット分析とモデル構築

EMG-EPN612データセットの探索的データ分析（EDA）とジェスチャー認識モデルの実装を提供するリポジトリです。

## 🌿 ブランチ構成

- **main/master**: EDA（探索的データ分析）
- **baseline**: 機械学習モデル実装（ベースラインモデル）

## 📊 概要

EMG-EPN612データセットは、Myo Armbandを使用して記録された筋電図（EMG）信号を含むジェスチャー認識用データセットです。本リポジトリでは、包括的なデータ分析スクリプトと詳細な分析結果を提供しています。

### データセット仕様
- **総ユーザー数**: 612名（トレーニング306名、テスト306名）
- **デバイス**: Myo Armband
- **サンプリング周波数**: 200 Hz
- **記録時間**: 1ジェスチャーあたり5秒
- **EMGチャンネル数**: 8チャンネル
- **ジェスチャークラス**: 6種類（fist, waveIn, waveOut, open, pinch, noGesture）

---

## 📁 リポジトリ構成

```
.
├── emg_eda.py                      # EDA分析スクリプト
├── eda_results/                    # 分析結果ディレクトリ
│   ├── EDA_REPORT.md              # 包括的な分析レポート（英語）
│   ├── emg_signals_8channels.png  # 8チャンネルEMG信号の可視化
│   ├── gesture_detection_timeline.png  # ジェスチャー検出タイムライン
│   ├── demographics_summary.png   # ユーザー属性の可視化
│   ├── emg_statistics_summary.png # EMG統計の可視化
│   ├── gesture_distribution.png   # ジェスチャー頻度分布
│   ├── demographics_stats.csv     # 詳細な属性統計
│   └── emg_stats.csv             # 詳細なEMG統計
├── .gitignore                     # Git無視ファイル（データセットを除外）
└── README.md                      # このファイル
```

---

## 💾 データセットのダウンロード

実際のデータセットファイル（`trainingJSON/`と`testingJSON/`）は、サイズの都合上このリポジトリには含まれていません。

データセットのダウンロード方法：
1. EMG-EPN612データセットを元のソースからダウンロード
2. ダウンロードした`trainingJSON/`と`testingJSON/`フォルダをルートディレクトリに配置

---

## 🔧 必要なライブラリ

```bash
pip install numpy pandas matplotlib seaborn
```

Python 3.7以上が必要です。

---

## 🚀 使用方法

1. データセットをルートディレクトリに配置
2. EDAスクリプトを実行：

```bash
python emg_eda.py
```

スクリプトの実行内容：
- データセット構造と属性の分析
- EMG信号統計の計算
- 可視化の生成
- 包括的なレポートの作成

すべての結果は`eda_results/`ディレクトリに保存されます。

---

## 📈 分析結果

### 1. EMG信号の可視化

#### 8チャンネルEMG信号
![EMG Signals](eda_results/emg_signals_8channels.png)

8つのEMGチャンネルの時系列信号を表示。各チャンネルの平均値と標準偏差も表示されています。

#### ジェスチャー検出タイムライン
![Gesture Timeline](eda_results/gesture_detection_timeline.png)

時間経過に伴うジェスチャー検出の推移を可視化。各ジェスチャーが異なる色で表現されています。

---

### 2. ユーザー属性分析

![Demographics](eda_results/demographics_summary.png)

#### 主な発見
- **年齢**: 平均24.3歳（範囲：18-54歳）
- **性別**: 男性70%、女性30%
- **利き手**: 右手95%、左手5%
- **民族**: 複数の民族グループが含まれる

⚠️ **注意**: データセットには若年成人への偏りがあり、性別にも不均衡が見られます。

---

### 3. EMG信号統計

![EMG Statistics](eda_results/emg_statistics_summary.png)

#### 全体統計
- **平均振幅**: -0.84 ± 0.76
- **標準偏差**: 14.10 ± 12.57
- **信号範囲**: 143.10 ± 79.80

#### チャンネル別の特徴

| チャンネル | 平均振幅 | 標準偏差 | 信号範囲 |
|-----------|---------|---------|---------|
| CH1 | -0.86 ± 0.15 | 6.92 ± 3.65 | 95.90 ± 47.78 |
| CH2 | -0.96 ± 0.22 | 12.46 ± 5.71 | 155.62 ± 60.44 |
| CH3 | -0.45 ± 1.29 | 28.99 ± 12.80 | 233.15 ± 52.58 |
| CH4 | -0.73 ± 1.60 | 27.53 ± 13.09 | 232.77 ± 38.98 |
| CH5 | -0.99 ± 0.18 | 16.25 ± 10.70 | 170.60 ± 58.48 |
| CH6 | -0.93 ± 0.20 | 9.01 ± 10.25 | 102.27 ± 58.88 |
| CH7 | -0.89 ± 0.16 | 6.58 ± 6.15 | 82.14 ± 56.64 |
| CH8 | -0.88 ± 0.15 | 5.06 ± 2.31 | 72.32 ± 33.12 |

**重要な発見**:
- **Channel 3と4**が最も高い変動性を示す（標準偏差 ~28-29）
- より動的な筋肉活動を捉えている可能性が高い
- **Channel 1, 7, 8**は比較的安定した信号

---

### 4. ジェスチャー分布

![Gesture Distribution](eda_results/gesture_distribution.png)

| ジェスチャー | サンプル数 | 割合 |
|-------------|-----------|------|
| No Gesture | 22,235 | 44.5% |
| Wave Out | 16,641 | 33.3% |
| Wave In | 3,666 | 7.3% |
| Open | 3,038 | 6.1% |
| Fist | 1,746 | 3.5% |
| Pinch | 101 | 0.2% |

⚠️ **クラス不均衡の問題**:
- "No Gesture"が44.5%で最多
- "Pinch"がわずか0.2%で極端に少ない
- 機械学習モデル訓練時には**クラス重み付け**や**データ拡張**が必要

---

## 🔍 EDAスクリプトの機能

1. **データセット概要**
   - ユーザー数と分布
   - 記録仕様

2. **属性分析**
   - 年齢、性別、利き手の分布
   - 民族的多様性

3. **EMG信号分析**
   - チャンネル別統計
   - 信号振幅と変動性
   - ジェスチャー頻度分布

4. **可視化**
   - 8チャンネルEMG時系列プロット
   - ジェスチャー検出タイムライン
   - 統計サマリープロット
   - 属性分布チャート

5. **統計レポート**
   - 詳細統計のCSVファイル
   - 包括的なMarkdownレポート

---

## 🤖 機械学習開発の推奨事項

### 1. 前処理
```python
# 推奨される前処理パイプライン
- バンドパスフィルタ（20-450 Hz）
- ノッチフィルタ（50/60 Hz電源ノイズ除去）
- 正規化/標準化
```

### 2. 特徴量エンジニアリング

**時間領域特徴量**:
- RMS（二乗平均平方根）
- MAV（平均絶対値）
- ZC（ゼロクロッシング）
- SSC（傾き符号変化）
- WL（波形長）

**周波数領域特徴量**:
- MNF（平均周波数）
- MDF（中央周波数）
- パワースペクトル密度

**時間-周波数特徴量**:
- ウェーブレット係数
- 短時間フーリエ変換（STFT）

### 3. モデル訓練

**推奨アルゴリズム**:
- **古典的機械学習**: SVM、Random Forest、k-NN
- **深層学習**: CNN、LSTM、CNN-LSTMハイブリッド
- **転移学習**: 事前学習モデルのEMGへの適用

**クラス不均衡への対処**:
```python
# クラス重み付け
from sklearn.utils.class_weight import compute_class_weight

# オーバーサンプリング（SMOTE等）
from imblearn.over_sampling import SMOTE

# データ拡張
- 時間シフト
- ノイズ注入
- スケーリング
```

### 4. 評価戦略

```python
# 推奨される評価指標
- 正解率（Accuracy）だけでなく
- 適合率（Precision）
- 再現率（Recall）
- F1スコア ← クラス不均衡に対して重要
- 混同行列（Confusion Matrix）
- クラス別性能評価
```

**クロスバリデーション**:
- 層化k分割交差検証（Stratified k-fold）を使用
- ユーザー独立評価 vs ユーザー依存評価

---

## 📊 データ品質評価

### ✅ 強み
- 大規模なサンプルサイズ（612ユーザー）
- 高品質な8チャンネルEMGデータ
- 標準化された記録プロトコル（200 Hz、5秒）
- 多様な人口統計学的表現
- 構造化されたJSON形式

### ⚠️ 制限事項
- 顕著なクラス不均衡（特にPinchジェスチャー: 0.2%）
- 性別の不均衡（男性70%、女性30%）
- 利き手の偏り（右手95%）
- 若年成人への年齢の偏り（平均24.3歳）
- 不明なジェスチャー状態の存在（5.0%）

### 💡 推奨対策
1. マイノリティクラスへのデータ拡張適用
2. 訓練/検証分割に層化サンプリングを使用
3. 過小表現ジェスチャーの追加データ収集を検討
4. 適切な評価指標の使用（F1スコア、バランス正解率）

---

## 🎯 潜在的な応用

このデータセットは以下の用途に適しています：

1. **ジェスチャー認識**: リアルタイム手ジェスチャー分類のための機械学習モデル訓練
2. **ヒューマンコンピュータインタラクション**: EMGベースの制御インターフェース開発
3. **義肢制御**: 筋電義肢の筋電制御に関する研究
4. **ユーザー認証**: 生体認証としてのEMG信号の探索
5. **信号処理研究**: EMG信号特性と前処理技術の研究

---

## 📝 ライセンス

元のEMG-EPN612データセットのライセンス条項を参照してください。

---

## 🤝 コントリビューション

コントリビューションを歓迎します！IssueやPull Requestをお気軽に提出してください。

---

## 📧 お問い合わせ

質問や問題がある場合は、GitHubでIssueを開いてください。

---

## 🤖 機械学習モデル（baselineブランチ）

`baseline`ブランチには、ジェスチャー認識のための機械学習モデルが実装されています。

### モデル構築の使用方法

```bash
# baselineブランチに切り替え
git checkout baseline

# 依存ライブラリをインストール
pip install -r requirements_ml.txt

# ベースラインモデルを訓練
python train_baseline.py --data_path . --max_users 20 --model_type random_forest
```

### 実装されているモデル

- **Random Forest**: アンサンブル学習ベースの分類器
- **SVM (Support Vector Machine)**: サポートベクターマシン
- **k-NN (k-Nearest Neighbors)**: 最近傍法

### 主な機能

1. **データローダー** (`src/data/data_loader.py`)
   - EMG-EPN612データセットの読み込み
   - ウィンドウベースのセグメンテーション
   - 層化サンプリング

2. **特徴量抽出** (`src/features/feature_extractor.py`)
   - 時間領域特徴量（MAV, RMS, ZC, SSC, WL等）
   - 周波数領域特徴量（MNF, MDF, パワースペクトル等）
   - バンドパス・ノッチフィルタ
   - 信号正規化

3. **ベースラインモデル** (`src/models/baseline_models.py`)
   - クラス不均衡への対処
   - モデル訓練と評価
   - 混同行列と性能比較の可視化

4. **訓練パイプライン** (`train_baseline.py`)
   - エンドツーエンドの訓練パイプライン
   - コマンドライン引数でのカスタマイズ
   - 結果の自動保存

---

## 🧠 深層学習モデル（CNN-LSTM）

### モデルアーキテクチャ

**CNN-LSTMハイブリッドモデル**を実装し、全データセット（306ユーザー、7,836訓練サンプル）でGPU訓練を実施しました。

#### 構成:
- **入力**: 8チャンネル × 200時間ステップ
- **CNNブロック**: 3層（32→64→128フィルタ）
  - 各層: Conv1D → BatchNorm → ReLU → MaxPool → Dropout
- **LSTM**: 双方向2層（隠れ層サイズ128）
- **全結合層**: 256 → 128 → 6クラス
- **パラメータ数**: 791,078個
- **デバイス**: NVIDIA GeForce RTX 5090 (CUDA 12.8)

### 訓練設定

```bash
python train_deep_learning.py --model_type cnn_lstm --epochs 50 --batch_size 64 --lr 0.001
```

- **エポック数**: 50
- **バッチサイズ**: 64
- **最適化手法**: Adam (lr=0.001, weight_decay=1e-4)
- **損失関数**: CrossEntropyLoss（クラスウェイト付き）
- **学習率スケジューラ**: ReduceLROnPlateau
- **データ分割**:
  - 訓練: 6,268サンプル
  - 検証: 1,568サンプル
  - テスト: 7,773サンプル

### 訓練結果

#### 最終精度
- **検証精度**: 70.15% (Epoch 2で達成)
- **テスト精度**: 72.08%

#### 訓練曲線
![Training Curves](results/cnn_lstm/training_curves.png)

**観察された現象**:
- Epoch 1-2: 検証精度が70%に急上昇
- Epoch 3以降: 検証精度が3-5%に急落し、その後も低迷
- 訓練精度は47.88%まで緩やかに上昇
- 明らかな**過学習**の兆候

---

### ⚠️ **重大な問題: モデルは実質的にジェスチャーを認識していない**

#### テストセット分類レポート

```
Classification Report (Test Set):
              precision    recall  f1-score   support

  No Gesture       0.72      1.00      0.84      5603
        Fist       0.00      0.00      0.00       258
     Wave In       0.00      0.00      0.00       149
    Wave Out       0.00      0.00      0.00      1493
        Open       0.00      0.00      0.00       270

    accuracy                           0.72      7773
   macro avg       0.14      0.20      0.17      7773
weighted avg       0.52      0.72      0.60      7773
```

#### 検証セット分類レポート

```
Classification Report (Validation Set):
              precision    recall  f1-score   support

  No Gesture       0.70      1.00      0.82      1100
        Fist       0.00      0.00      0.00        42
     Wave In       0.00      0.00      0.00        61
    Wave Out       0.00      0.00      0.00       298
        Open       0.00      0.00      0.00        67

    accuracy                           0.70      1568
   macro avg       0.14      0.20      0.16      1568
weighted avg       0.49      0.70      0.58      1568
```

#### 混同行列
![Confusion Matrix - Test](results/cnn_lstm/confusion_matrix_test.png)
![Confusion Matrix - Validation](results/cnn_lstm/confusion_matrix_validation.png)

---

### 📊 問題分析

#### 1. **多数派クラス予測（Majority Class Prediction）**

モデルは**すべてのサンプルを「No Gesture」クラスに分類**しています：

- **No Gesture**: Recall = 1.00（すべてのサンプルをNo Gestureと予測）
- **その他のジェスチャー**: Recall = 0.00（全く認識できていない）

#### 2. **見かけ上の高精度の落とし穴**

72.08%という精度は一見良好に見えますが、これは：
- テストセットの72.1%がNo Gestureであるため
- 単純に「すべてNo Gesture」と予測しているだけ
- **実用的なジェスチャー認識モデルとしては完全に機能していない**

#### 3. **根本原因**

##### a) **極端なクラス不均衡**
```
訓練セット:
  No Gesture: 70.4% (5,514サンプル)
  Fist:        2.8% (221サンプル)
  Wave In:     3.1% (240サンプル)
  Wave Out:   19.4% (1,524サンプル)
  Open:        4.3% (336サンプル)
  Pinch:       0.0% (1サンプル)  ← 極端に少ない
```

##### b) **クラスウェイトの限界**
- CrossEntropyLossにクラスウェイトを適用したが不十分
- 重み付けだけでは70:30や80:20のような極端な不均衡には対処しきれない

##### c) **過学習の進行**
- Epoch 2以降、検証精度が急落
- モデルは訓練データの多数派クラスに特化してしまった

---

### 🔧 改善策

#### 1. **データレベルの対策**

##### a) リサンプリング
```python
# アンダーサンプリング: No Gestureクラスを削減
# オーバーサンプリング: マイノリティクラスを増加

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# 組み合わせ戦略
smote_tomek = SMOTETomek(random_state=42)
X_resampled, y_resampled = smote_tomek.fit_resample(X_train, y_train)
```

##### b) データ拡張の強化
```python
# 現在の拡張: ノイズ、時間シフト、スケーリング
# 追加可能な拡張:
- 時間ワーピング（Time Warping）
- マグニチュード変更
- チャンネルシャッフル
- ミックスアップ（Mixup）
```

##### c) Pinchクラスの除外検討
```python
# Pinchクラス（1サンプルのみ）を除外して5クラス分類に変更
classes_to_keep = [0, 1, 2, 3, 4]  # Pinch (5) を除外
```

#### 2. **損失関数の改善**

##### a) Focal Loss
```python
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

##### b) Class-Balanced Loss
```python
from pytorch_metric_learning import losses

# Effective Number of Samples を考慮
beta = 0.9999
effective_num = 1.0 - np.power(beta, samples_per_class)
weights = (1.0 - beta) / effective_num
weights = weights / weights.sum() * num_classes
```

#### 3. **モデルアーキテクチャの改善**

```python
# より深いネットワーク
# Attentionメカニズムの追加
# ResNet風のスキップ接続
```

#### 4. **訓練戦略の改善**

##### a) 2段階訓練
```python
# Stage 1: 全クラスを均等にサンプリングして事前訓練
# Stage 2: 元の分布でファインチューニング
```

##### b) Curriculum Learning
```python
# 簡単なサンプル（No Gesture vs その他）から始める
# 徐々に難しいタスク（全6クラス分類）に移行
```

##### c) Early Stopping の改善
```python
# 検証精度だけでなくF1スコアやバランス精度も監視
# マイノリティクラスの性能も考慮
```

#### 5. **評価指標の改善**

```python
# 精度だけでなく以下も追加監視:
- バランス精度（Balanced Accuracy）
- マクロ平均F1スコア（Macro-averaged F1）
- クラス別Recall（特にマイノリティクラス）
- Cohen's Kappa係数
```

---

### 📁 保存されたファイル

```
results/cnn_lstm/
├── best_model.pth                    # ベストモデル（Epoch 2）
├── training_curves.png               # 訓練・検証の損失/精度曲線
├── confusion_matrix_validation.png   # 検証セット混同行列
├── confusion_matrix_test.png         # テストセット混同行列
└── tensorboard/                      # TensorBoardログ
```

### 使用方法

#### モデルの訓練
```bash
# CNN-LSTMモデルを訓練
python train_deep_learning.py --model_type cnn_lstm --epochs 50 --batch_size 64 --lr 0.001

# 他のモデルタイプ
python train_deep_learning.py --model_type cnn           # Simple CNN
python train_deep_learning.py --model_type attention_lstm # Attention LSTM
```

#### TensorBoardでの可視化
```bash
tensorboard --logdir results/cnn_lstm/tensorboard
```

---

### 🎯 今後の方向性

**現状**: モデルは多数派クラス予測に陥っており、実用的なジェスチャー認識は不可能

**次のステップ**:
1. **最優先**: リサンプリング戦略の実装（SMOTE + アンダーサンプリング）
2. Focal Lossへの変更
3. データ拡張の大幅強化
4. Pinchクラスの除外（5クラス分類への変更）
5. モデルアーキテクチャの見直し（Attention機構の追加等）

**目標精度**:
- 単純な精度: 70-80%（現在72%だが無意味）
- **重要**: 各ジェスチャーのRecallが最低40%以上
- マクロ平均F1スコア: 0.60以上

---

**最終更新**: 2025-10-23
**分析ツール**: Python 3.x (NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn, PyTorch)
**データセット**: EMG-EPN612 (612ユーザー、8チャンネル、200 Hz)
**GPUデバイス**: NVIDIA GeForce RTX 5090 (CUDA 12.8)
