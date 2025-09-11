#!/usr/bin/env python3
"""
Large TSP instance experiments with route visualization
Tests 30, 50, 100 node instances with OR-Tools and MIP solvers
"""

import sys
import os
import time
import json
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

sys.path.append('../src')
sys.path.append('../../../src')
from tsp_utils import TSPDataExtractor
from tsp_ortools import solve_tsp_with_ortools
from tsp_mip import solve_tsp_with_mip
from tsp_visualization import visualize_tsp_route, create_route_comparison_plot


def main():
    print("🚀 大規模TSP実験をDatabricks MLflowに記録します...")
    print("実験対象: 30, 50ノードのインスタンス")
    
    # Databricks MLflow設定
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/data_science/z_ogai/tsp-experiments")
    
    # 1. 大規模TSPインスタンスを作成
    print("\n1. 大規模TSPインスタンスを作成中...")
    vrp_file = "../../tai75a/data/tai75a.vrp"
    extractor = TSPDataExtractor(vrp_file)
    
    tsp_sizes = [30, 50]
    tsp_instances = {}
    
    for size in tsp_sizes:
        print(f"  TSP{size}インスタンスを作成中...")
        tsp_data = extractor.extract_tsp_subset(size, include_depot=True)
        tsp_instances[f"tsp{size}"] = tsp_data
        print(f"    TSP{size}: {tsp_data['dimension']} nodes")
        print(f"    Distance matrix shape: {tsp_data['distance_matrix'].shape}")
    
    # 2. OR-Toolsで解く
    print("\n2. OR-Toolsで大規模インスタンスを解いています...")
    ortools_results = {}
    
    for name, tsp_data in tsp_instances.items():
        print(f"  {name}をOR-Toolsで解いています...")
        
        # 大きなインスタンスには長めの時間制限を設定
        size = tsp_data['dimension']
        if size <= 30:
            time_limit = 120
        elif size <= 50:
            time_limit = 300  # 5分
        else:
            time_limit = 600  # 10分
            
        print(f"    Time limit: {time_limit}秒")
        
        start_time = time.time()
        result = solve_tsp_with_ortools(tsp_data, time_limit=time_limit)
        elapsed_time = time.time() - start_time
        
        ortools_results[name] = result
        
        cost = result.get('solution_cost', 'N/A')
        solve_time = result.get('solve_time_seconds', elapsed_time)
        print(f"    コスト: {cost}, 時間: {solve_time:.2f}秒")
    
    # 3. MIPで解く
    print("\n3. MIPで大規模インスタンスを解いています...")
    mip_results = {}
    
    for name, tsp_data in tsp_instances.items():
        print(f"  {name}をMIPで解いています...")
        
        # 大きなインスタンスには長めの時間制限を設定
        size = tsp_data['dimension']
        if size <= 30:
            time_limit = 120
        elif size <= 50:
            time_limit = 300  # 5分
        else:
            time_limit = 600  # 10分
            
        print(f"    Time limit: {time_limit}秒")
        
        start_time = time.time()
        result = solve_tsp_with_mip(tsp_data, time_limit=time_limit)
        elapsed_time = time.time() - start_time
        
        mip_results[name] = result
        
        cost = result.get('solution_cost', 'N/A')
        solve_time = result.get('solve_time_seconds', elapsed_time)
        print(f"    コスト: {cost}, 時間: {solve_time:.2f}秒")
    
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
    
    # 5. ルート可視化を作成
    print("\n5. ルート可視化を作成中...")
    
    visualization_files = []
    
    for name, tsp_data in tsp_instances.items():
        ortools_result = ortools_results[name]
        mip_result = mip_results[name]
        
        # 各ソルバーの結果があるかチェック
        ortools_tour = ortools_result.get('tour', [])
        mip_tour = mip_result.get('tour', [])
        ortools_cost = ortools_result.get('solution_cost', 0)
        mip_cost = mip_result.get('solution_cost', 0)
        
        if ortools_tour or mip_tour:
            # 比較可視化を作成
            comparison_file = f"{name}_route_comparison.png"
            create_route_comparison_plot(
                tsp_data, 
                ortools_tour, 
                mip_tour,
                ortools_cost,
                mip_cost,
                comparison_file
            )
            visualization_files.append(comparison_file)
            print(f"    {comparison_file} を作成しました")
        
        # 個別のルート可視化
        if ortools_tour:
            ortools_file = f"{name}_ortools_route.png"
            visualize_tsp_route(
                tsp_data,
                ortools_tour,
                f"OR-Tools Solution - {name.upper()} (Cost: {ortools_cost:.1f})",
                ortools_file
            )
            visualization_files.append(ortools_file)
        
        if mip_tour:
            mip_file = f"{name}_mip_route.png"
            visualize_tsp_route(
                tsp_data,
                mip_tour,
                f"MIP Solution - {name.upper()} (Cost: {mip_cost:.1f})",
                mip_file
            )
            visualization_files.append(mip_file)
    
    # 6. 統計可視化を作成
    print("6. 統計可視化を作成中...")
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
    ax1.set_title('Solution Quality Comparison (Large Instances)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 対数スケールで見やすく
    
    # 時間比較
    ax2.plot(sizes, ortools_times, 'o-', label='OR-Tools', color='blue', linewidth=2, markersize=8)
    ax2.plot(sizes, mip_times, 's-', label='MIP', color='red', linewidth=2, markersize=8)
    ax2.set_xlabel('TSP Instance Size')
    ax2.set_ylabel('Solve Time (seconds)')
    ax2.set_title('Computation Time Comparison (Large Instances)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # 対数スケールで見やすく
    
    # 棒グラフ - コスト
    x = np.arange(len(sizes))
    width = 0.35
    ax3.bar(x - width/2, ortools_costs, width, label='OR-Tools', color='blue', alpha=0.7)
    ax3.bar(x + width/2, mip_costs, width, label='MIP', color='red', alpha=0.7)
    ax3.set_xlabel('TSP Instance Size')
    ax3.set_ylabel('Solution Cost')
    ax3.set_title('Solution Cost Comparison (Large Instances)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'TSP{s}' for s in sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 棒グラフ - 時間
    ax4.bar(x - width/2, ortools_times, width, label='OR-Tools', color='blue', alpha=0.7)
    ax4.bar(x + width/2, mip_times, width, label='MIP', color='red', alpha=0.7)
    ax4.set_xlabel('TSP Instance Size')
    ax4.set_ylabel('Solve Time (seconds)')
    ax4.set_title('Computation Time Comparison (Large Instances)')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'TSP{s}' for s in sizes])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    stats_plot_file = 'large_tsp_comparison_stats.png'
    plt.savefig(stats_plot_file, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    visualization_files.append(stats_plot_file)
    
    # 7. Databricks MLflowに記録
    print("\n7. Databricks MLflowに記録中...")
    
    # OR-Tools結果を記録
    for name, result in ortools_results.items():
        with mlflow.start_run(run_name=f"ortools_{name}_large"):
            # パラメータをログ
            mlflow.log_param("solver_type", "OR-Tools")
            mlflow.log_param("instance_name", name)
            mlflow.log_param("instance_size", result['instance_info']['dimension'])
            mlflow.log_param("experiment_type", "large_instances")
            
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
            result_file = f"{name}_ortools_large_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            mlflow.log_artifact(result_file)
            os.remove(result_file)
            
            # ルート可視化をアーティファクトとして記録
            ortools_route_file = f"{name}_ortools_route.png"
            if ortools_route_file in visualization_files:
                mlflow.log_artifact(ortools_route_file)
    
    print("  OR-Tools結果をDatabricks MLflowに記録しました")
    
    # MIP結果を記録
    for name, result in mip_results.items():
        with mlflow.start_run(run_name=f"mip_{name}_large"):
            # パラメータをログ
            mlflow.log_param("solver_type", "MIP")
            mlflow.log_param("instance_name", name)
            mlflow.log_param("instance_size", result['instance_info']['dimension'])
            mlflow.log_param("experiment_type", "large_instances")
            
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
            result_file = f"{name}_mip_large_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            mlflow.log_artifact(result_file)
            os.remove(result_file)
            
            # ルート可視化をアーティファクトとして記録
            mip_route_file = f"{name}_mip_route.png"
            if mip_route_file in visualization_files:
                mlflow.log_artifact(mip_route_file)
    
    print("  MIP結果をDatabricks MLflowに記録しました")
    
    # 比較サマリーを記録
    with mlflow.start_run(run_name="large_tsp_comparison_summary"):
        mlflow.log_param("experiment_type", "Large_TSP_Comparison")
        mlflow.log_param("num_instances", len(tsp_instances))
        mlflow.log_param("instance_sizes", str(tsp_sizes))
        mlflow.log_param("max_instance_size", max(tsp_sizes))
        
        # 集計メトリクス
        ortools_wins = sum(1 for data in comparison_data if data['winner'] == 'OR-Tools')
        mip_wins = sum(1 for data in comparison_data if data['winner'] == 'MIP')
        ties = sum(1 for data in comparison_data if data['winner'] == 'Tie')
        
        mlflow.log_metric("ortools_wins", ortools_wins)
        mlflow.log_metric("mip_wins", mip_wins)
        mlflow.log_metric("ties", ties)
        
        # 平均パフォーマンス
        avg_ortools_time = np.mean([d['ortools_time'] for d in comparison_data])
        avg_mip_time = np.mean([d['mip_time'] for d in comparison_data])
        mlflow.log_metric("avg_ortools_time", avg_ortools_time)
        mlflow.log_metric("avg_mip_time", avg_mip_time)
        
        # 比較データ
        summary_file = "large_comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        mlflow.log_artifact(summary_file)
        os.remove(summary_file)
        
        # すべての可視化ファイルをアーティファクトとして記録
        for viz_file in visualization_files:
            if os.path.exists(viz_file):
                mlflow.log_artifact(viz_file)
    
    print("  比較サマリーと可視化をDatabricks MLflowに記録しました")
    
    # 8. ファイルクリーンアップ
    for viz_file in visualization_files:
        if os.path.exists(viz_file):
            os.remove(viz_file)
    
    # 9. 結果の表示
    print("\n" + "="*80)
    print("大規模TSP Solver Comparison Results")
    print("="*80)
    print(f"{'Instance':<10} {'OR-Tools Cost':<15} {'MIP Cost':<15} {'OR-Tools Time':<15} {'MIP Time':<15} {'Winner':<10}")
    print("-"*80)
    
    for data in comparison_data:
        ot_cost = data['ortools_cost']
        mp_cost = data['mip_cost']
        ot_cost_str = f"{ot_cost:.1f}" if ot_cost != float('inf') else "N/A"
        mp_cost_str = f"{mp_cost:.1f}" if mp_cost != float('inf') else "N/A"
        
        print(f"{data['instance']:<10} {ot_cost_str:<15} {mp_cost_str:<15} {data['ortools_time']:<15.2f} {data['mip_time']:<15.2f} {data['winner']:<10}")
    
    print(f"\nOR-Tools勝利: {ortools_wins}")
    print(f"MIP勝利: {mip_wins}")
    print(f"引き分け: {ties}")
    
    print(f"\nOR-Tools平均時間: {avg_ortools_time:.2f}秒")
    print(f"MIP平均時間: {avg_mip_time:.2f}秒")
    
    print("\n✅ 大規模TSP実験が完了し、Databricks MLflowに記録されました！")
    print("📊 Databricksで実験結果を確認できます: /Shared/data_science/z_ogai/tsp-large-experiments")
    print("🎨 ルート可視化もアーティファクトとして保存されています")


if __name__ == "__main__":
    main()