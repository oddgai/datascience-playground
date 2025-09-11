#!/usr/bin/env python3
"""
VRP実験結果をDatabricks MLflowに記録するスクリプト

使用方法:
    python log_to_mlflow.py instances/f-n45-k4/results/experiment_results.json
    python log_to_mlflow.py instances/f-n135-k7/results/experiment_results.json
"""

import json
import sys
import os
from datetime import datetime
from typing import Dict, Any
import mlflow
try:
    from .visualization import create_comparison_visualization
except ImportError:
    # When running as script, use absolute import
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from visualization import create_comparison_visualization


def log_experiment_to_mlflow(experiment_json_path: str):
    """MLflowに実験結果を記録"""
    
    # 1. Databricks MLflow設定
    print("Databricks MLflowに接続中...")
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/data_science/z_ogai/vrp-instances")
    
    # 2. JSONファイルから実験結果を読み込み
    print(f"実験結果を読み込み中: {experiment_json_path}")
    with open(experiment_json_path, 'r', encoding='utf-8') as f:
        exp_results = json.load(f)
    
    # 3. Run名と説明文を生成
    instance_name = exp_results["instance_info"]["name"].lower()
    experiment_id = exp_results["experiment_id"].split('_')[-1]  # exp003 -> 003
    run_name = f"{instance_name}_{experiment_id}"
    
    # 日本語で簡潔な説明
    gap = exp_results["gap_percentage"]
    if gap == 0:
        performance = "最適解と完全一致"
    elif gap <= 5:
        performance = f"優秀な解（ギャップ {gap:.2f}%）"
    else:
        performance = f"良好な解（ギャップ {gap:.2f}%）"
    
    description = (
        f"OR-tools VRPソルバーによる{exp_results['instance_info']['name']}インスタンスの解法実験。"
        f"Decimal.quantize()による正確な距離計算を実装。{performance}を達成。"
    )
    
    print(f"MLflow Runを開始: {run_name}")
    
    # 4. MLflowにパラメータ、メトリクス、アーティファクトを記録
    with mlflow.start_run(run_name=run_name, description=description):
        
        # パラメータ記録
        mlflow.log_param("model_type", exp_results["model_type"])
        mlflow.log_param("instance_name", exp_results["instance_info"]["name"])
        mlflow.log_param("dimension", exp_results["instance_info"]["dimension"])
        mlflow.log_param("num_vehicles", exp_results["instance_info"]["num_vehicles"])
        mlflow.log_param("capacity", exp_results["instance_info"]["capacity"])
        mlflow.log_param("distance_calculation", exp_results["preprocessing"]["distance_calculation"])
        
        # ソルバー設定
        for key, value in exp_results["model_params"].items():
            mlflow.log_param(f"solver_{key}", value)
        
        # メトリクス記録
        mlflow.log_metric("optimal_cost", exp_results["optimal_cost"])
        mlflow.log_metric("solution_cost", exp_results["solution_cost"])
        mlflow.log_metric("gap_percentage", exp_results["gap_percentage"])
        mlflow.log_metric("solve_time_seconds", exp_results["solve_time_seconds"])
        mlflow.log_metric("num_routes", exp_results["num_routes"])
        
        # 追加メトリクス
        mlflow.log_metric("cost_efficiency", exp_results["optimal_cost"] / exp_results["solution_cost"])
        
        # ベンチマーク比較（f-n45-k4のみ）
        if "benchmark_comparison" in exp_results:
            mlflow.log_metric("vs_int_improvement", exp_results["benchmark_comparison"]["vs_int_method"]["improvement"])
            mlflow.log_metric("vs_round_improvement", exp_results["benchmark_comparison"]["vs_round_method"]["improvement"])
        
        # スケーラビリティ分析（f-n135-k7のみ）
        if "scalability_analysis" in exp_results:
            mlflow.log_metric("node_scaling_factor", exp_results["scalability_analysis"]["vs_f_n45_k4"]["node_scaling_factor"])
            mlflow.log_metric("cost_scaling_factor", exp_results["scalability_analysis"]["vs_f_n45_k4"]["cost_scaling_factor"])
        
        # タグ設定
        mlflow.set_tag("problem_type", "CVRP")
        mlflow.set_tag("solver", "OR-tools")
        mlflow.set_tag("distance_method", "decimal_quantize")
        mlflow.set_tag("experiment_date", datetime.now().strftime("%Y-%m-%d"))
        
        if exp_results["gap_percentage"] == 0:
            mlflow.set_tag("performance", "optimal")
        elif exp_results["gap_percentage"] <= 5:
            mlflow.set_tag("performance", "excellent")
        elif exp_results["gap_percentage"] <= 10:
            mlflow.set_tag("performance", "good")
        else:
            mlflow.set_tag("performance", "fair")
        
        # 上下比較可視化を作成
        print("上下比較可視化を作成中...")
        try:
            # パスから実験情報を抽出
            path_parts = experiment_json_path.split('/')
            instance_name = path_parts[1]  # f-n135-k7 or f-n45-k4
            
            #実験IDを取得
            if len(path_parts) >= 4 and path_parts[3] != "experiment_results.json":
                # 例: instances/f-n135-k7/results/exp002/experiment_results.json
                exp_id = path_parts[3]  # exp002
            else:
                # 例: instances/f-n45-k4/results/experiment_results.json
                # experiment_results.jsonから実験IDを推定
                exp_id = exp_results["experiment_id"].split('_')[-1]  # f-n45-k4_exp003 -> exp003
            
            experiment_dir = f"instances/{instance_name}/experiments/{exp_id}"
            vrp_file = f"instances/{instance_name}/data/{instance_name}.vrp"
            sol_file = f"instances/{instance_name}/data/{instance_name}.sol"
            
            # 上下比較可視化を作成
            viz_path = create_comparison_visualization(
                experiment_dir=experiment_dir,
                experiment_id=exp_id,
                vrp_file=vrp_file,
                sol_file=sol_file,
                our_cost=exp_results["solution_cost"],
                optimal_cost=exp_results["optimal_cost"]
            )
            
            # 可視化をアーティファクトとして保存
            mlflow.log_artifact(viz_path, "visualizations")
            print(f"✅ 上下比較可視化を作成しました: {viz_path}")
            
        except Exception as e:
            print(f"⚠️  上下比較可視化の作成に失敗しました: {e}")
            # 可視化が失敗してもMLflowの記録は続行
        
        # アーティファクトとしてJSONファイルを保存
        mlflow.log_artifact(experiment_json_path, "experiment_results")
        
        print("✅ MLflowへの記録が完了しました")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


def main():
    if len(sys.argv) != 2:
        print("使用方法: python log_to_mlflow.py <experiment_results.json_path>")
        print("例:")
        print("  python log_to_mlflow.py instances/f-n45-k4/results/experiment_results.json")
        print("  python log_to_mlflow.py instances/f-n135-k7/results/experiment_results.json")
        sys.exit(1)
    
    experiment_json_path = sys.argv[1]
    
    if not os.path.exists(experiment_json_path):
        print(f"❌ ファイルが見つかりません: {experiment_json_path}")
        sys.exit(1)
    
    try:
        log_experiment_to_mlflow(experiment_json_path)
        print("🎉 MLflow記録が正常に完了しました！")
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()