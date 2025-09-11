#!/usr/bin/env python3
"""
MLflowの古いrunを削除するスクリプト

使用方法:
    python scripts/delete_mlflow_runs.py
"""

import mlflow

def delete_old_runs():
    """VRPインスタンス実験の古いrunを削除"""
    
    # Databricks MLflowに接続
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/data_science/z_ogai/vrp-instances")
    
    # 実験のすべてのrunを取得
    experiment = mlflow.get_experiment_by_name("/Shared/data_science/z_ogai/vrp-instances")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    print(f"見つかったruns: {len(runs)}個")
    
    # 各runを削除
    for index, run in runs.iterrows():
        run_id = run['run_id']
        run_name = run.get('tags.mlflow.runName', 'Unknown')
        print(f"削除中: {run_name} (ID: {run_id})")
        
        try:
            mlflow.delete_run(run_id)
            print(f"✅ 削除成功: {run_name}")
        except Exception as e:
            print(f"❌ 削除失敗: {run_name} - {e}")
    
    print("🎉 古いrunsの削除が完了しました！")

if __name__ == "__main__":
    delete_old_runs()