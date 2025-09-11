# exp001: OR-toolsによるVRPソルバー実装（F-n135-k7）

## 概要

Google OR-toolsを使用してF-n135-k7インスタンス（135ノード、7車両）のCapacitated Vehicle Routing Problem (CVRP)を解く実験
距離計算でDecimal.quantize(ROUND_HALF_UP)を使用し、TSPLIBの標準に準拠

## 使用技術

- Python 3.12
- Google OR-tools
- 最適化手法: OR-toolsのルーティングソルバー
- 距離計算: Decimal.quantize(ROUND_HALF_UP)による数学的四捨五入（TSPLIBのnint()と同等）

## 実験パラメータ

| パラメータ | 値 |
|-----------|-----|
| ソルバー | OR-tools Routing |
| 探索戦略 | PATH_CHEAPEST_ARC |
| メタヒューリスティック | GUIDED_LOCAL_SEARCH |
| 時間制限 | 60秒 |

## インスタンス詳細

| 項目 | 値 |
|------|-----|
| ノード数 | 135（デポ1 + 顧客134） |
| 車両数 | 7 |
| 車両容量 | 2210 |
| 最適解コスト | 1162 |

## ベースとなる実験

- f-n45-k4/experiments/exp003: Decimal.quantize()版で最適解と完全一致を達成

## 実装内容

1. **utils.py**: VRPデータの読み込みと前処理
   - .vrpファイルのパース
   - 距離行列の計算（Decimal.quantize()使用）
   - データ構造の準備

2. **exp001.ipynb**: メインの実験コード
   - OR-toolsのルーティングモデル構築
   - 容量制約の設定
   - 最適化の実行
   - 結果の可視化

3. **config.yaml**: 実験パラメータの管理

## 期待される成果

- 最適解（コスト: 1162）に近い解の発見
- 大規模インスタンスでのOR-toolsの性能評価
- Decimal.quantize()による正確な距離計算の検証

## 次の実験への改善案

- より長い計算時間での探索
- 異なる探索戦略の試行
- 並列計算の活用