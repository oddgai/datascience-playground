#!/usr/bin/env python3
"""
Very Large TSP instance experiments with route visualization
Tests 100, 150 node instances from tai100a and tai150a with OR-Tools and MIP solvers
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
    print("実験対象: 100ノード (tai100a), 150ノード (tai150a)")
    
    # Databricks MLflow設定
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/data_science/z_ogai/tsp-experiments")
    
    # 1. 大規模TSPインスタンスを作成
    print("\n1. 大規模TSPインスタンスを作成中...")
    
    # tai100aから100ノードTSP、tai150aから150ノードTSPを作成
    instances_config = [
        {"name": "tsp100", "size": 100, "vrp_file": "../../tai100a/data/tai100a.vrp"},
        {"name": "tsp150", "size": 150, "vrp_file": "../../tai150a/data/tai150a.vrp"}
    ]
    
    tsp_instances = {}
    
    for config in instances_config:
        print(f"  {config['name']}インスタンスを作成中...")
        print(f"    ソースファイル: {config['vrp_file']}")
        
        extractor = TSPDataExtractor(config['vrp_file'])
        tsp_data = extractor.extract_tsp_subset(config['size'], include_depot=True)
        tsp_instances[config['name']] = tsp_data
        
        print(f"    {config['name']}: {tsp_data['dimension']} nodes")
        print(f"    Distance matrix shape: {tsp_data['distance_matrix'].shape}")
    
    # 2. OR-Toolsで解く
    print("\n2. OR-Toolsで大規模インスタンスを解いています...")
    ortools_results = {}
    
    for name, tsp_data in tsp_instances.items():
        size = tsp_data['dimension']
        # 大規模インスタンス用に時間制限を調整
        if size <= 100:
            time_limit = 600  # 10分
        else:
            time_limit = 1200  # 20分
            
        print(f"  {name}をOR-Toolsで解いています...")
        print(f"    Time limit: {time_limit}秒")
        
        start_time = time.time()
        result = solve_tsp_with_ortools(tsp_data, time_limit=time_limit)
        end_time = time.time()
        
        ortools_results[name] = result
        cost = result.get('solution_cost', 'N/A')
        solve_time = result.get('solve_time_seconds', end_time - start_time)
        
        print(f"    コスト: {cost}, 時間: {solve_time:.2f}秒")

    # 3. MIPで解く  
    print("\n3. MIPで大規模インスタンスを解いています...")
    mip_results = {}
    
    for name, tsp_data in tsp_instances.items():
        size = tsp_data['dimension']
        # 大規模インスタンス用に時間制限を調整
        if size <= 100:
            time_limit = 1800  # 30分
        else:
            time_limit = 3600  # 60分
            
        print(f"  {name}をMIPで解いています...")
        print(f"    Time limit: {time_limit}秒")
        
        start_time = time.time()
        result = solve_tsp_with_mip(tsp_data, time_limit=time_limit)
        end_time = time.time()
        
        mip_results[name] = result
        cost = result.get('solution_cost', 'N/A')
        solve_time = result.get('solve_time_seconds', end_time - start_time)
        
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
            'size': tsp_instances[name]['dimension'],
            'ortools_cost': ortools_cost,
            'mip_cost': mip_cost,
            'ortools_time': ortools_time,
            'mip_time': mip_time,
            'winner': winner
        })
    
    # 結果を出力
    print("\n" + "="*80)
    print("大規模TSP実験結果")
    print("="*80)
    print(f"{'Instance':<10} {'Size':<6} {'OR-Tools Cost':<15} {'MIP Cost':<15} {'OR-Tools Time':<15} {'MIP Time':<15} {'Winner':<10}")
    print("-"*100)
    
    for data in comparison_data:
        print(f"{data['instance']:<10} {data['size']:<6} {data['ortools_cost']:<15.1f} {data['mip_cost']:<15.1f} {data['ortools_time']:<15.2f} {data['mip_time']:<15.2f} {data['winner']:<10}")

    # 5. ルート可視化を作成
    print("\n5. ルート可視化を作成中...")
    visualization_files = {}
    
    for name, tsp_data in tsp_instances.items():
        ortools_result = ortools_results[name]
        mip_result = mip_results[name]
        
        ortools_route = ortools_result.get('tour', [])
        mip_route = mip_result.get('tour', [])
        ortools_cost = ortools_result.get('solution_cost', float('inf'))
        mip_cost = mip_result.get('solution_cost', float('inf'))
        
        if ortools_route and mip_route:
            # 比較プロットを作成
            comparison_file = f"{name}_route_comparison.png"
            try:
                create_route_comparison_plot(
                    tsp_data, 
                    ortools_route, 
                    mip_route,
                    ortools_cost,
                    mip_cost,
                    save_path=comparison_file
                )
                visualization_files[f"{name}_comparison"] = comparison_file
                print(f"  {name}比較プロット作成: {comparison_file}")
            except Exception as e:
                print(f"  {name}比較プロット作成エラー: {e}")
        
        # 個別のルート可視化
        if ortools_route:
            ortools_file = f"{name}_ortools_route.png"
            try:
                visualize_tsp_route(
                    tsp_data, 
                    ortools_route, 
                    title=f"OR-Tools TSP Solution - {name.upper()} (Cost: {ortools_cost:.1f})",
                    save_path=ortools_file
                )
                visualization_files[f"{name}_ortools"] = ortools_file
                print(f"  {name} OR-Toolsルート作成: {ortools_file}")
            except Exception as e:
                print(f"  {name} OR-Toolsルート作成エラー: {e}")
        
        if mip_route:
            mip_file = f"{name}_mip_route.png"
            try:
                visualize_tsp_route(
                    tsp_data, 
                    mip_route, 
                    title=f"MIP TSP Solution - {name.upper()} (Cost: {mip_cost:.1f})",
                    save_path=mip_file
                )
                visualization_files[f"{name}_mip"] = mip_file
                print(f"  {name} MIPルート作成: {mip_file}")
            except Exception as e:
                print(f"  {name} MIPルート作成エラー: {e}")

    # 6. MLflowに結果を記録
    print("\n6. MLflowに結果を記録中...")
    
    # OR-Tools結果を記録
    for name, result in ortools_results.items():
        with mlflow.start_run(run_name=f"ortools_{name}_large"):
            # パラメータを記録
            mlflow.log_param("solver_type", "OR-Tools")
            mlflow.log_param("instance_name", name)
            mlflow.log_param("instance_size", result['instance_info']['dimension'])
            mlflow.log_param("experiment_type", "very_large_tsp")
            
            if 'model_params' in result:
                for param, value in result['model_params'].items():
                    mlflow.log_param(param, value)
            
            # メトリクスを記録
            if 'solution_cost' in result:
                mlflow.log_metric("solution_cost", result['solution_cost'])
            
            if 'solve_time_seconds' in result:
                mlflow.log_metric("solve_time_seconds", result['solve_time_seconds'])
            
            if 'is_optimal' in result:
                mlflow.log_metric("is_optimal", 1 if result['is_optimal'] else 0)
            
            # 結果JSONをアーティファクトとして記録
            result_file = f"{name}_ortools_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            mlflow.log_artifact(result_file)
            os.remove(result_file)
            
            # 可視化ファイルを記録
            if f"{name}_ortools" in visualization_files:
                mlflow.log_artifact(visualization_files[f"{name}_ortools"])
            
            if f"{name}_comparison" in visualization_files:
                mlflow.log_artifact(visualization_files[f"{name}_comparison"])
    
    print("OR-Tools結果をMLflowに記録完了")
    
    # MIP結果を記録
    for name, result in mip_results.items():
        with mlflow.start_run(run_name=f"mip_{name}_large"):
            # パラメータを記録
            mlflow.log_param("solver_type", "MIP")
            mlflow.log_param("instance_name", name)
            mlflow.log_param("instance_size", result['instance_info']['dimension'])
            mlflow.log_param("experiment_type", "very_large_tsp")
            
            if 'model_params' in result:
                for param, value in result['model_params'].items():
                    mlflow.log_param(param, value)
            
            # メトリクスを記録
            if 'solution_cost' in result:
                mlflow.log_metric("solution_cost", result['solution_cost'])
            
            if 'solve_time_seconds' in result:
                mlflow.log_metric("solve_time_seconds", result['solve_time_seconds'])
            
            if 'is_optimal' in result:
                mlflow.log_metric("is_optimal", 1 if result['is_optimal'] else 0)
            
            # 結果JSONをアーティファクトとして記録
            result_file = f"{name}_mip_result.json"
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            mlflow.log_artifact(result_file)
            os.remove(result_file)
            
            # 可視化ファイルを記録
            if f"{name}_mip" in visualization_files:
                mlflow.log_artifact(visualization_files[f"{name}_mip"])
    
    print("MIP結果をMLflowに記録完了")
    
    # 比較サマリーを記録
    with mlflow.start_run(run_name="very_large_tsp_comparison_summary"):
        mlflow.log_param("experiment_type", "Very_Large_TSP_Comparison")
        mlflow.log_param("num_instances", len(tsp_instances))
        mlflow.log_param("instance_sizes", "100,150")
        
        # 集計メトリクス
        ortools_wins = sum(1 for data in comparison_data if data['winner'] == 'OR-Tools')
        mip_wins = sum(1 for data in comparison_data if data['winner'] == 'MIP')
        ties = sum(1 for data in comparison_data if data['winner'] == 'Tie')
        
        mlflow.log_metric("ortools_wins", ortools_wins)
        mlflow.log_metric("mip_wins", mip_wins)
        mlflow.log_metric("ties", ties)
        
        # 平均メトリクス
        valid_ortools_costs = [d['ortools_cost'] for d in comparison_data if d['ortools_cost'] != float('inf')]
        valid_mip_costs = [d['mip_cost'] for d in comparison_data if d['mip_cost'] != float('inf')]
        
        if valid_ortools_costs:
            mlflow.log_metric("avg_ortools_cost", np.mean(valid_ortools_costs))
            mlflow.log_metric("avg_ortools_time", np.mean([d['ortools_time'] for d in comparison_data]))
        
        if valid_mip_costs:
            mlflow.log_metric("avg_mip_cost", np.mean(valid_mip_costs))
            mlflow.log_metric("avg_mip_time", np.mean([d['mip_time'] for d in comparison_data]))
        
        # 比較データをアーティファクトとして記録
        comparison_file = "very_large_tsp_comparison_summary.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        mlflow.log_artifact(comparison_file)
        os.remove(comparison_file)
    
    print("比較サマリーをMLflowに記録完了")
    
    # 可視化ファイルをクリーンアップ
    for file_path in visualization_files.values():
        if os.path.exists(file_path):
            os.remove(file_path)
    
    print("\n✅ 大規模TSP実験が完了しました！")
    print(f"MLflow実験: /Shared/data_science/z_ogai/tsp-experiments")
    print("すべての結果、可視化、比較データがDatabricks MLflowに記録されました。")


if __name__ == "__main__":
    main()