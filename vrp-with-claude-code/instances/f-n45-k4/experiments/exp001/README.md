# exp001: OR-toolsによるVRPソルバー実装

## 概要

Google OR-toolsを使用してCapacitated Vehicle Routing Problem (CVRP)を解く実験

## 使用技術

- Python 3.12
- Google OR-tools
- 最適化手法: OR-toolsのルーティングソルバー

## 実験パラメータ

| パラメータ | 値 |
|-----------|-----|
| ソルバー | OR-tools Routing |
| 探索戦略 | FIRST_SOLUTION_CHEAPEST_ARC |
| メタヒューリスティック | GUIDED_LOCAL_SEARCH |
| 時間制限 | 30秒 |

## ベースとなる実験

- 初回実験のため、ベースなし

## 実装内容

1. **utils.py**: VRPデータの読み込みと前処理
   - .vrpファイルのパース
   - 距離行列の計算
   - データ構造の準備

2. **exp001.ipynb**: メインの実験コード
   - OR-toolsのルーティングモデル構築
   - 容量制約の設定
   - 最適化の実行
   - 結果の可視化

3. **config.yaml**: 実験パラメータの管理

## 期待される成果

- 最適解（コスト: 724）に近い解の発見
- 計算時間の測定
- 解の品質評価

## 次の実験への改善案

- 異なる探索戦略の試行
- メタヒューリスティックのパラメータ調整
- 初期解の改善