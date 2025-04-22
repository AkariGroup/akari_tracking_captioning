# akari_tracking_captioning

OAK-Dカメラを使用して人物を検出・追跡し、その人物の外観や行動を自動的にキャプション化するシステムです。

## 概要

このシステムは以下の機能を提供します：

- OAK-Dカメラを使用した人物の検出と追跡
- 検出した人物の外観特徴の自動記述
- 人物の行動のリアルタイム認識とキャプション化
- 時系列での行動ログの自動要約
- CSV形式でのログ保存

## 必要な環境
- AKARI(本アプリを実行)
- Nvidia GPUを搭載したPC([local_vlm_server](https://github.com/AkariGroup/local_vlm_server) を実行)

## セットアップ
1. submoduleの初期化

```bash
git submodule update --init --recursive
```

2. 仮想環境の作成

```bash
python -m venv venv
```

3. 仮想環境のアクティベート
```bash
. venv/bin/activate
```

4. 必要なライブラリのインストール

```bash
pip install -r requirements.txt
```

5. Nvidia GPUを搭載したPCで[local_vlm_server](https://github.com/AkariGroup/local_vlm_server)をREADMEに沿ってセットアップ。

## 使用方法
1. 外部PCでVLMサーバーを起動。

2. 本アプリを起動

```bash
python main.py
```

以下のオプションが利用可能：  
   - `-r, --robot_coordinate`: カメラ座標系からロボット座標系への変換を有効にする
   - `--vlm_host`: ローカルVLMサーバーのホスト名を指定します（デフォルト: 127.0.0.1）
   - `--vlm_port`: ローカルVLMサーバーのポート番号を指定します（デフォルト: 10020）

例：
```bash
python main.py -r --vlm_host 192.168.1.10 --vlm_port 10020
```

## 動作の仕組み

1. OAK-Dカメラで人物を検出し、追跡（OakdTrackingYoloクラス）
2. 検出された人物ごとに以下の処理を行う：
   - 初回検出時：外観の特徴をVLMで分析
   - 追跡中：行動をVLMで分析し続ける
3. トラッキングが外れた人物の記録を自動的に要約
4. ログをCSVファイルに保存（log/ディレクトリ）

## ログファイル

ログファイルは `log/` ディレクトリに保存され、以下の形式のCSVファイルとして出力される：

```
ID,開始時刻,終了時刻,年齢,性別,外観,行動要約
```


