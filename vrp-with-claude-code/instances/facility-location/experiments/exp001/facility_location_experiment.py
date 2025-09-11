#!/usr/bin/env python3
"""
Facility Location Problem experiment with tai100a data using MIP solver
Records results to Databricks MLflow
"""

import sys
import os
import time
import json
import mlflow
import numpy as np

# Add the src path to sys.path
sys.path.append('../../src')
sys.path.append('../../../../src')

from facility_utils import FacilityLocationDataExtractor, load_facility_location_data
from facility_mip import solve_facility_location_with_mip
from facility_visualization import visualize_facility_location, create_facility_analysis_plot


def main():
    print("🏢 施設配置問題実験をDatabricks MLflowに記録します...")
    print("実験対象: tai100a (99ヶ所の候補地から10ヶ所の施設を選択)")
    
    # Databricks MLflow設定
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/data_science/z_ogai/facility-location-experiments")
    
    # 1. 施設配置問題データを作成
    print("\\n1. 施設配置問題データを作成中...")
    
    vrp_file = "../../../tai100a/data/tai100a.vrp"
    print(f"VRPファイル: {vrp_file}")
    
    # データ抽出（depotを除外して99ヶ所を対象にする）
    extractor = FacilityLocationDataExtractor(vrp_file)
    fl_data = extractor.extract_facility_location_data(exclude_depot=True)
    
    print(f"施設配置問題データ:")
    print(extractor.get_problem_summary(fl_data))
    
    # 2. MIPで施設配置問題を解く
    print("\\n2. MIPで施設配置問題を解いています...")
    
    # 設定パラメータ
    num_facilities = 10    # 設置する施設数
    time_limit = 600       # 10分の時間制限
    
    print(f"施設数: {num_facilities}")
    print(f"時間制限: {time_limit}秒")
    
    start_time = time.time()
    result = solve_facility_location_with_mip(fl_data, num_facilities=num_facilities, time_limit=time_limit)
    end_time = time.time()
    
    total_experiment_time = end_time - start_time
    
    facility_locations = result.get('facility_locations', [])
    assignments = result.get('assignments', {})
    solution_cost = result.get('solution_cost', float('inf'))
    optimization_gap = result.get('optimization_gap')
    is_optimal = result.get('is_optimal', False)
    solver_used = result.get('model_params', {}).get('solver', 'Unknown')
    
    print(f"\\n解決状況: {'最適解' if is_optimal else '実行可能解'}") 
    print(f"施設設置場所: {facility_locations}")
    print(f"総コスト: {solution_cost}")
    if optimization_gap is not None:
        print(f"最適化ギャップ: {optimization_gap:.4f}")
    print(f"使用ソルバー: {solver_used}")
    print(f"総実験時間: {total_experiment_time:.2f}秒")
    
    # 3. 可視化を作成
    print("\\n3. 可視化を作成中...")
    visualization_files = {}
    
    if facility_locations and assignments:
        try:
            # 基本的な施設配置図
            main_viz_file = visualize_facility_location(
                fl_data, 
                facility_locations, 
                assignments,
                title=f"Facility Location Solution - {num_facilities} Facilities\\nTotal Cost: {solution_cost:.2f}",
                save_path="facility_location_solution.png"
            )
            visualization_files["main_solution"] = main_viz_file
            print(f"  メイン可視化作成: {main_viz_file}")
            
            # 詳細分析図
            analysis_viz_file = create_facility_analysis_plot(
                fl_data,
                facility_locations,
                assignments,
                solution_cost,
                save_path="facility_analysis.png"
            )
            visualization_files["detailed_analysis"] = analysis_viz_file
            print(f"  分析可視化作成: {analysis_viz_file}")
            
        except Exception as e:
            print(f"  可視化作成エラー: {e}")
    
    # 4. MLflowに結果を記録
    print("\\n4. MLflowに結果を記録中...")
    
    with mlflow.start_run(run_name=f"facility_location_tai100a_{num_facilities}fac"):
        # パラメータを記録
        mlflow.log_param("problem_type", "Facility_Location")
        mlflow.log_param("solver_type", "MIP")
        mlflow.log_param("instance_name", "tai100a")
        mlflow.log_param("num_locations", fl_data['num_locations'])
        mlflow.log_param("num_facilities", num_facilities)
        mlflow.log_param("time_limit_seconds", time_limit)
        mlflow.log_param("exclude_depot", fl_data['exclude_depot'])
        mlflow.log_param("total_demand", fl_data['total_demand'])
        
        if 'model_params' in result:
            for param, value in result['model_params'].items():
                mlflow.log_param(f"model_{param}", value)
        
        # メトリクスを記録
        mlflow.log_metric("solution_cost", solution_cost)
        mlflow.log_metric("solve_time_seconds", result.get('solve_time_seconds', 0))
        mlflow.log_metric("total_experiment_time_seconds", total_experiment_time)
        mlflow.log_metric("is_optimal", 1 if is_optimal else 0)
        
        if optimization_gap is not None:
            mlflow.log_metric("optimization_gap", optimization_gap)
        
        # 施設配置場所を記録（文字列として）
        facility_locations_str = ','.join(map(str, facility_locations))
        mlflow.log_param("facility_locations", facility_locations_str)
        
        # 解の品質指標
        solution_quality = result.get('solution_quality', {})
        if solution_quality:
            mlflow.log_metric("solution_valid", 1 if solution_quality.get('valid', False) else 0)
            mlflow.log_metric("cost_verification_match", 1 if solution_quality.get('cost_match', False) else 0)
            mlflow.log_metric("num_facilities_correct", 1 if solution_quality.get('num_facilities_correct', False) else 0)
            mlflow.log_metric("all_demands_assigned", 1 if solution_quality.get('all_demands_assigned', False) else 0)
        
        # 施設別統計
        if assignments:
            facility_loads = {}
            facility_costs = {}
            distance_matrix = fl_data['distance_matrix']
            demands = fl_data['demands']
            
            for demand_point, facility in assignments.items():
                if facility not in facility_loads:
                    facility_loads[facility] = 0
                    facility_costs[facility] = 0
                
                demand = demands[demand_point]
                distance = distance_matrix[demand_point][facility]
                cost = demand * distance
                
                facility_loads[facility] += demand
                facility_costs[facility] += cost
            
            # 施設統計をログ
            mlflow.log_metric("avg_facility_load", np.mean(list(facility_loads.values())))
            mlflow.log_metric("max_facility_load", np.max(list(facility_loads.values())))
            mlflow.log_metric("min_facility_load", np.min(list(facility_loads.values())))
            mlflow.log_metric("avg_facility_cost", np.mean(list(facility_costs.values())))
            mlflow.log_metric("max_facility_cost", np.max(list(facility_costs.values())))
        
        # 結果JSONをアーティファクトとして記録
        result_file = "facility_location_result.json"
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        mlflow.log_artifact(result_file)
        os.remove(result_file)
        
        # 問題データサマリーを記録
        problem_summary_file = "problem_summary.txt"
        with open(problem_summary_file, 'w') as f:
            f.write(extractor.get_problem_summary(fl_data))
        mlflow.log_artifact(problem_summary_file)
        os.remove(problem_summary_file)
        
        # 可視化ファイルを記録
        for viz_name, viz_file in visualization_files.items():
            if os.path.exists(viz_file):
                mlflow.log_artifact(viz_file)
                os.remove(viz_file)
                print(f"  {viz_name}をMLflowにアップロード: {viz_file}")
    
    print("\\nMLflowに結果記録完了")
    
    # 5. 結果サマリー
    print("\\n" + "="*70)
    print("🏢 施設配置問題実験結果サマリー")
    print("="*70)
    print(f"インスタンス: {fl_data['name']}")
    print(f"候補地数: {fl_data['num_locations']}")
    print(f"設置施設数: {num_facilities}")
    print(f"総需要: {fl_data['total_demand']:.1f}")
    print(f"使用ソルバー: {solver_used}")
    print(f"解の状況: {'最適解' if is_optimal else '実行可能解'}")
    print(f"総コスト: {solution_cost:.2f}")
    if optimization_gap is not None:
        print(f"最適化ギャップ: {optimization_gap:.4f}")
    print(f"解決時間: {result.get('solve_time_seconds', 0):.2f}秒")
    print(f"総実験時間: {total_experiment_time:.2f}秒")
    print(f"施設設置場所: {facility_locations}")
    
    if assignments:
        print(f"\\n施設負荷分散:")
        facility_loads = {}
        for demand_point, facility in assignments.items():
            if facility not in facility_loads:
                facility_loads[facility] = 0
            facility_loads[facility] += fl_data['demands'][demand_point]
        
        for facility in sorted(facility_loads.keys()):
            load = facility_loads[facility]
            print(f"  施設 {facility}: 需要負荷 {load:.1f}")
    
    print("\\n✅ 施設配置問題実験が完了しました！")
    print(f"MLflow実験: /Shared/data_science/z_ogai/facility-location-experiments")
    print("すべての結果、可視化、分析データがDatabricks MLflowに記録されました。")


if __name__ == "__main__":
    main()