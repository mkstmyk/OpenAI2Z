# OpenAI to Z Challenge - Checkpoint 1 & 2

Amazon 流域の Sentinel-2 L2A COG (GeoTIFF) を取得し、NDVI 異常域を自動抽出して OpenAI GPT-4o-mini に説明させる考古学リモートセンシングツール。

## 🚀 機能

### ✅ 実装済み機能
- **Sentinel-2 データダウンロード**: AWS S3 からの自動取得
- **NDVI 計算**: 植生指標の算出と可視化
- **異常域抽出**: 閾値ベースの自動検出
- **OpenAI 分析**: GPT-4o-mini による考古学的解釈
- **Checkpoint 1**: 複数データソース読み込み
- **Checkpoint 2**: 新規サイト発見（アルゴリズム検出 + 歴史的テキスト + 既知サイト比較）

### 🔧 デバッグ機能
- **OpenAI API スキップ**: デバッグ中にクレジット消費を回避
- **ダミーデータ生成**: 実データが利用できない場合の代替
- **段階的実行**: 重い処理のスキップオプション

## 📊 データソース状況

### ✅ 正常動作
- **Sentinel-2**: AWS S3 からの直接ダウンロード
- **サンプル考古学データ**: ローカル生成（実データ代替）
- **サンプル標高データ**: ローカル生成（SRTM代替）
- **植生データ**: Sentinel-2 派生（GEDI代替）

### ⚠️ 制限事項
- **TerraBrasilis**: URL解決エラー（代替データ使用）
- **OpenTopography SRTM**: API 404エラー（代替データ使用）
- **GEDI L2A**: 直接アクセス困難（Sentinel-2派生データ使用）

### 🔄 改善予定
- **NASA Earthdata API**: より信頼性の高いSRTMデータ
- **OpenStreetMap**: 考古学サイトの公開データ
- **UNESCO**: 世界遺産サイトデータ

## 🛠️ セットアップ

### 1. 環境準備
```bash
# 仮想環境作成
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# 依存関係インストール
pip install -r requirements.txt
```

### 2. OpenAI API キー設定
```bash
# 環境変数設定
export OPENAI_API_KEY="your-api-key-here"

# または .env ファイル作成
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### 3. AWS CLI 設定（Sentinel-2 データ用）
```bash
# AWS CLI インストール（未インストールの場合）
pip install awscli

# 設定（認証情報不要 - パブリックデータ）
aws configure set default.s3.signature_version s3v4
```

## 🚀 使用方法

### 基本実行
```bash
python openai_to_z_checkpoint.py
```

### デバッグモード（推奨）
```python
# openai_to_z_checkpoint.py の main() 関数内で
DEBUG_MODE = True  # OpenAI API スキップ
```

### 本番モード
```python
# openai_to_z_checkpoint.py の main() 関数内で
DEBUG_MODE = False  # 実際の OpenAI API 呼び出し
```

## 📁 出力ファイル

### メイン分析
- `data_dir/footprints.json`: 検出された異常域
- `data_dir/ndvi_map.png`: NDVI 可視化
- `openai_log.json`: OpenAI 分析ログ

### Checkpoint 1
- `data_dir/archaeological_sites.geojson`: 考古学サイトデータ
- `data_dir/srtm_elevation.tif`: 標高データ
- `data_dir/vegetation_data.json`: 植生データ

### Checkpoint 2
- `data_dir/checkpoint2_candidates.geojson`: アルゴリズム検出結果
- `data_dir/historical_extracts.json`: 歴史的テキスト抽出
- `data_dir/site_comparison.json`: 既知サイト比較結果

## 🔍 トラブルシューティング

### よくある問題

1. **OpenAI API エラー**
   ```bash
   # 環境変数確認
   echo $OPENAI_API_KEY
   
   # デバッグモードでテスト
   DEBUG_MODE = True
   ```

2. **Sentinel-2 ダウンロードエラー**
   ```bash
   # AWS CLI 確認
   aws --version
   
   # ネットワーク接続確認
   curl -I https://sentinel-s2-l2a.s3.amazonaws.com
   ```

3. **メモリ不足エラー**
   ```python
   # 重い処理をスキップ
   skip_heavy = True
   ```

### データソース問題

1. **実データが利用できない場合**
   - 自動的にサンプルデータが生成されます
   - 処理は継続され、機能テストが可能です

2. **特定のデータソースが必要な場合**
   - 手動でデータをダウンロード
   - `data_dir/` に配置
   - スクリプトを再実行

## 📈 パフォーマンス

### 実行時間（目安）
- **デバッグモード**: 2-3分
- **本番モード**: 5-10分（API呼び出し時間含む）
- **重い処理スキップ**: 1-2分

### メモリ使用量
- **基本処理**: 500MB-1GB
- **重い処理**: 2-4GB
- **推奨**: 8GB以上

## 🎯 Checkpoint 要件対応状況

### Checkpoint 1 ✅
- [x] 複数の独立データソース読み込み
- [x] 5つ以上の異常フットプリント生成
- [x] データセットIDとOpenAIプロンプトのログ
- [x] 再現可能なスクリプト

### Checkpoint 2 ✅
- [x] アルゴリズム検出（Hough変換）
- [x] 歴史的テキスト抽出（GPT使用）
- [x] 既知考古学サイトとの比較

## 🤝 貢献

1. 実データソースの追加
2. アルゴリズムの改善
3. エラーハンドリングの強化
4. ドキュメントの改善

## 📄 ライセンス

MIT License - オープンソースプロジェクトとして公開

## 🔗 参考資料

- [OpenAI to Z Challenge](https://openai.com/blog/openai-to-z-challenge)
- [Starter Pack](documents/starter-pack-openai-to-z-challenge.txt)
- [Checkpoints Guide](documents/checkpoints-openai-to-z-challenge.txt) 