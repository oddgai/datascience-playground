#!/usr/bin/env python3
"""
TSP実験結果をDatabricks MLflowに記録するスクリプト
"""

import sys
import os
import time
import json
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

sys.path.append('src')
sys.path.append('../../src')
from tsp_utils import TSPDataExtractor
from tsp_ortools import solve_tsp_with_ortools
from tsp_mip import solve_tsp_with_mip

def main():
    print("TSP実験をDatabricks MLflowに記録します...")
    
    # Databricks MLflow設定
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/data_science/z_ogai/tsp-experiments")
    
    # 1. TSPインスタンスを作成
    print("1. TSPインスタンスを作成中...")
    vrp_file = "../tai75a/data/tai75a.vrp"
    extractor = TSPDataExtractor(vrp_file)
    
    tsp_sizes = [10, 15, 20]
    tsp_instances = {}
    
    for size in tsp_sizes:
        tsp_data = extractor.extract_tsp_subset(size, include_depot=True)
        tsp_instances[f"tsp{size}"] = tsp_data
        print(f"  TSP{size}: {tsp_data['dimension']} nodes")
    
    # 2. OR-Toolsで解く
    print("\n2. OR-Toolsで解いています...")
    ortools_results = {}
    
    for name, tsp_data in tsp_instances.items():
        print(f"  {name}を解いています...")
        result = solve_tsp_with_ortools(tsp_data, time_limit=60)
        ortools_results[name] = result
        print(f"    コスト: {result.get('solution_cost', 'N/A')}")
    
    # 3. MIPで解く
    print("\n3. MIPで解いています...")
    mip_results = {}
    
    for name, tsp_data in tsp_instances.items():
        print(f"  {name}を解いています...")
        result = solve_tsp_with_mip(tsp_data, time_limit=60)
        mip_results[name] = result
        print(f"    コスト: {result.get('solution_cost', 'N/A')}")
    
    # 4. 比較データを作成
    print("\n4. 比較データを作成中...")
    comparison_data = []
    
    for name in tsp_instances.keys():
        ortools_cost = ortools_results[name].get('solution_cost', float('inf'))
        mip_cost = mip_results[name].get('solution_cost', float('inf'))
        ortools_time = ortools_results[name].get('solve_time_seconds', 0)
        mip_time = mip_results[name].get('solve_time_seconds', 0)
        
        if ortools_cost == float('inf') and mip_cost == float('inf'):
            winner = "None"
        elif ortools_cost == float('inf'):
            winner = "MIP"
        elif mip_cost == float('inf'):
            winner = "OR-Tools"
        elif ortools_cost < mip_cost:
            winner = "OR-Tools"
        elif mip_cost < ortools_cost:
            winner = "MIP"
        else:
            winner = "Tie"
        
        comparison_data.append({
            'instance': name,
            'size': int(name.replace('tsp', '')),
            'ortools_cost': ortools_cost,
            'mip_cost': mip_cost,
            'ortools_time': ortools_time,
            'mip_time': mip_time,
            'winner': winner
        })
    
    # 5. 可視化を作成
    print("5. 可視化を作成中...")
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    sizes = [data['size'] for data in comparison_data]
    ortools_costs = [data['ortools_cost'] for data in comparison_data]
    mip_costs = [data['mip_cost'] for data in comparison_data]
    ortools_times = [data['ortools_time'] for data in comparison_data]
    mip_times = [data['mip_time'] for data in comparison_data]
    
    # コスト比較
    ax1.plot(sizes, ortools_costs, 'o-', label='OR-Tools', color='blue', linewidth=2, markersize=8)
    ax1.plot(sizes, mip_costs, 's-', label='MIP', color='red', linewidth=2, markersize=8)
    ax1.set_xlabel('TSP Instance Size')
    ax1.set_ylabel('Solution Cost')
    ax1.set_title('Solution Quality Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 時間比較
    ax2.plot(sizes, ortools_times, 'o-', label='OR-Tools', color='blue', linewidth=2, markersize=8)
    ax2.plot(sizes, mip_times, 's-', label='MIP', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('TSP Instance Size')
    ax2.set_ylabel('Solve Time (seconds)')
    ax2.set_title('Computation Time Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 棒グラフ - コスト
    x = np.arange(len(sizes))
    width = 0.35
    ax3.bar(x - width/2, ortools_costs, width, label='OR-Tools', color='blue', alpha=0.7)
    ax3.bar(x + width/2, mip_costs, width, label='MIP', color='red', alpha=0.7)
    ax3.set_xlabel('TSP Instance Size')
    ax3.set_ylabel('Solution Cost')
    ax3.set_title('Solution Cost Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'TSP{s}' for s in sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 棒グラフ - 時間
    ax4.bar(x - width/2, ortools_times, width, label='OR-Tools', color='blue', alpha=0.7)
    ax4.bar(x + width/2, mip_times, width, label='MIP', color='red', alpha=0.7)
    ax4.set_xlabel('TSP Instance Size')
    ax4.set_ylabel('Solve Time (seconds)')
    ax4.set_title('Computation Time Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'TSP{s}' for s in sizes])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsp_comparison.png', dpi=80, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 6. Databricks MLflowに記録
    print("\n6. Databricks MLflowに記録中...")
    
    # OR-Tools結果を記録
    for name, result in ortools_results.items():
        with mlflow.start_run(run_name=f"ortools_{name}"):
            # パラメータをログ
            mlflow.log_param("solver_type", "OR-Tools")
            mlflow.log_param("instance_name", name)
            mlflow.log_param("instance_size", result['instance_info']['dimension'])
            
            if 'model_params' in result:
                for param, value in result['model_params'].items():
                    mlflow.log_param(param, value)
            
            # メトリクスをログ
            if 'solution_cost' in result:
                mlflow.log_metric("solution_cost", result['solution_cost'])
            
            if 'solve_time_seconds' in result:
                mlflow.log_metric("solve_time_seconds", result['solve_time_seconds'])
            
            if 'is_optimal' in result:
                mlflow.log_metric("is_optimal", 1 if result['is_optimal'] else 0)
            
            # 結果ファイルをアーティファクトとして記録
            with open(f"{name}_ortools_result.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
            mlflow.log_artifact(f"{name}_ortools_result.json")
            os.remove(f"{name}_ortools_result.json")
    
    print("  OR-Tools結果をDatabricks MLflowに記録しました")
    
    # MIP結果を記録
    for name, result in mip_results.items():
        with mlflow.start_run(run_name=f"mip_{name}"):
            # パラメータをログ
            mlflow.log_param("solver_type", "MIP")
            mlflow.log_param("instance_name", name)
            mlflow.log_param("instance_size", result['instance_info']['dimension'])
            
            if 'model_params' in result:
                for param, value in result['model_params'].items():
                    mlflow.log_param(param, value)
            
            # メトリクスをログ
            if 'solution_cost' in result:
                mlflow.log_metric("solution_cost", result['solution_cost'])
            
            if 'solve_time_seconds' in result:
                mlflow.log_metric("solve_time_seconds", result['solve_time_seconds'])
            
            if 'is_optimal' in result:
                mlflow.log_metric("is_optimal", 1 if result['is_optimal'] else 0)
            
            # 結果ファイルをアーティファクトとして記録
            with open(f"{name}_mip_result.json", 'w') as f:
                json.dump(result, f, indent=2, default=str)
            mlflow.log_artifact(f"{name}_mip_result.json")
            os.remove(f"{name}_mip_result.json")
    
    print("  MIP結果をDatabricks MLflowに記録しました")
    
    # 比較サマリーを記録
    with mlflow.start_run(run_name="tsp_comparison_summary"):
        mlflow.log_param("experiment_type", "TSP_Comparison")
        mlflow.log_param("num_instances", len(tsp_instances))
        mlflow.log_param("instance_sizes", str(tsp_sizes))
        
        # 集計メトリクス
        ortools_wins = sum(1 for data in comparison_data if data['winner'] == 'OR-Tools')
        mip_wins = sum(1 for data in comparison_data if data['winner'] == 'MIP')
        ties = sum(1 for data in comparison_data if data['winner'] == 'Tie')
        
        mlflow.log_metric("ortools_wins", ortools_wins)
        mlflow.log_metric("mip_wins", mip_wins)
        mlflow.log_metric("ties", ties)
        
        # 比較データ
        with open("comparison_summary.json", 'w') as f:
            json.dump(comparison_data, f, indent=2)
        mlflow.log_artifact("comparison_summary.json")
        os.remove("comparison_summary.json")
        
        # 比較プロット
        mlflow.log_artifact("tsp_comparison.png")
        os.remove("tsp_comparison.png")
    
    print("  比較サマリーをDatabricks MLflowに記録しました")
    
    # 7. 結果の表示
    print("\n" + "="*80)
    print("TSP Solver Comparison Results")
    print("="*80)
    print(f"{'Instance':<10} {'OR-Tools Cost':<15} {'MIP Cost':<15} {'OR-Tools Time':<15} {'MIP Time':<15} {'Winner':<10}")
    print("-"*80)
    
    for data in comparison_data:
        print(f"{data['instance']:<10} {data['ortools_cost']:<15.1f} {data['mip_cost']:<15.1f} {data['ortools_time']:<15.2f} {data['mip_time']:<15.2f} {data['winner']:<10}")
    
    print(f"\nOR-Tools勝利: {ortools_wins}")
    print(f"MIP勝利: {mip_wins}")
    print(f"引き分け: {ties}")
    
    print("\n✅ TSP実験が完了し、Databricks MLflowに記録されました！")
    print("📊 Databricksで実験結果を確認できます: /Shared/data_science/z_ogai/tsp-experiments")

if __name__ == "__main__":
    main()