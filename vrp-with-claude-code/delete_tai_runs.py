#!/usr/bin/env python3
"""
MLflowからtai関連の実験ランを削除するスクリプト
"""

import os
import mlflow
from mlflow.tracking import MlflowClient

# Databricks MLflowの設定
os.environ["DATABRICKS_HOST"] = "https://dbc-55810bf1-184f.cloud.databricks.com"
os.environ["DATABRICKS_TOKEN"] = "YOUR_DATABRICKS_TOKEN_HERE"

# MLflowクライアントの初期化
client = MlflowClient()

# 実験IDを指定
experiment_id = "4297944460811272"

print("🗑️ MLflowからtai関連の実験ランを削除中...")

try:
    # 実験内の全ランを取得
    runs = client.search_runs(experiment_ids=[experiment_id])
    
    tai_runs = []
    for run in runs:
        run_name = run.info.run_name or ""
        if any(instance in run_name.lower() for instance in ["tai75a", "tai100a", "tai150a", "tai385"]):
            tai_runs.append(run)
    
    print(f"📊 削除対象: {len(tai_runs)} 個のtai関連ラン")
    
    # tai関連のランを削除
    for run in tai_runs:
        print(f"  🗑️ 削除中: {run.info.run_name} (ID: {run.info.run_id})")
        client.delete_run(run.info.run_id)
    
    print(f"✅ {len(tai_runs)} 個のtai関連ランを削除完了！")
    
except Exception as e:
    print(f"❌ エラーが発生しました: {e}")
    
print("🎉 削除処理が完了しました。")