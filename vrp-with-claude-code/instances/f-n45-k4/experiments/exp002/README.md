# exp002: OR-toolsによるVRPソルバー実装（四捨五入版）

## 概要

Google OR-toolsを使用してCapacitated Vehicle Routing Problem (CVRP)を解く実験
距離計算でround()を使用し、exp001のint()切り捨てによる影響を検証

## 使用技術

- Python 3.12
- Google OR-tools
- 最適化手法: OR-toolsのルーティングソルバー
- 距離計算: round()による四捨五入（exp001はint()による切り捨て）

## 実験パラメータ

| パラメータ | 値 |
|-----------|-----|
| ソルバー | OR-tools Routing |
| 探索戦略 | FIRST_SOLUTION_CHEAPEST_ARC |
| メタヒューリスティック | GUIDED_LOCAL_SEARCH |
| 時間制限 | 30秒 |

## ベースとなる実験

- exp001: OR-toolsベースライン（int()による距離切り捨て版）

## 実装内容

1. **utils.py**: VRPデータの読み込みと前処理
   - .vrpファイルのパース
   - 距離行列の計算（round()使用）
   - データ構造の準備

2. **exp002.ipynb**: メインの実験コード
   - OR-toolsのルーティングモデル構築
   - 容量制約の設定
   - 最適化の実行
   - 結果の可視化

3. **config.yaml**: 実験パラメータの管理

## 期待される成果

- exp001より正確な距離計算による適正なコスト
- 最適解（コスト: 724）との適切な比較
- int()切り捨ての影響の定量化

## 次の実験への改善案

- より高精度な距離計算手法の検討
- 異なる探索戦略の試行
- メタヒューリスティックのパラメータ調整