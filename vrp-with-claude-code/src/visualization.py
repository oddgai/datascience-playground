"""
VRP解の可視化モジュール
"""

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from typing import List, Tuple, Dict
import os
import japanize_matplotlib
try:
    from .utils import VRPDataReader
except ImportError:
    # When running as script, use absolute import
    from utils import VRPDataReader

def visualize_vrp_solution(vrp_file: str, solution_routes: List[List[int]], 
                          output_path: str, title: str = "VRP Solution") -> str:
    """
    VRP解を可視化してPNGファイルとして保存
    
    Args:
        vrp_file: VRPインスタンスファイルのパス
        solution_routes: 解のルート（例：[[0,5,3,0], [0,2,4,1,0]]）
        output_path: 出力PNGファイルのパス
        title: グラフのタイトル
        
    Returns:
        保存されたファイルのパス
    """
    
    # VRPデータを読み込み
    vrp_data = VRPDataReader(vrp_file)
    vrp_data.parse()  # データを解析
    
    # node_coordsから座標マッピングを作成
    # VRPファイル: ノード1がデポット、2-45がお客様
    # ルート表現: ノード0がデポット、2-45がお客様のノード番号そのまま
    def get_coord(route_node_id):
        if route_node_id == 0:
            vrp_node_id = 1  # depot
        else:
            vrp_node_id = route_node_id  # customer nodes keep same ID
        return vrp_data.node_coords[vrp_node_id]
    
    depot_coord = get_coord(0)  # ルートのノード0 = VRPファイルのノード1（デポット）
    
    # プロット設定
    plt.figure(figsize=(12, 8))
    plt.style.use('default')
    
    # カラーマップを設定（車両ごとに異なる色）
    colors = plt.cm.Set1(np.linspace(0, 1, len(solution_routes)))
    
    # デポットをプロット（大きな黒い四角）
    plt.scatter(depot_coord[0], depot_coord[1], c='black', s=200, marker='s', 
                label='Depot', zorder=5, edgecolors='white', linewidth=2)
    
    # お客様をプロット（小さな灰色の円）
    customer_x = []
    customer_y = []
    for vrp_node_id in sorted(vrp_data.node_coords.keys()):
        if vrp_node_id != 1:  # デポット以外
            coord = vrp_data.node_coords[vrp_node_id]
            customer_x.append(coord[0])
            customer_y.append(coord[1])
    
    plt.scatter(customer_x, customer_y, c='lightgray', s=50, marker='o', 
                label='Customers', zorder=3, edgecolors='black', linewidth=0.5)
    
    
    # 各車両のルートをプロット
    for vehicle_id, route in enumerate(solution_routes):
        if len(route) <= 2:  # デポットのみの場合（0-0）はスキップ
            continue
            
        color = colors[vehicle_id]
        route_coords = [get_coord(node) for node in route]
        
        # ルートを線で結ぶ
        route_x = [coord[0] for coord in route_coords]
        route_y = [coord[1] for coord in route_coords]
        
        plt.plot(route_x, route_y, color=color, linewidth=2, alpha=0.8,
                label=f'Vehicle {vehicle_id + 1}', zorder=2)
        
        # ルート上の点を強調
        for coord in route_coords[1:-1]:  # デポットを除く
            plt.scatter(coord[0], coord[1], color=color, s=80, marker='o', zorder=4,
                       edgecolors='white', linewidth=1)
    
    # グラフの設定
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('X座標', fontsize=12)
    plt.ylabel('Y座標', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # レイアウトを調整
    plt.tight_layout()
    
    # ファイルを保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=80, bbox_inches='tight', facecolor='white')
    plt.close()  # メモリリークを防ぐ
    
    return output_path


def extract_routes_from_notebook_output(notebook_path: str) -> List[List[int]]:
    """
    Jupyterノートブックの出力からルート情報を抽出
    
    Args:
        notebook_path: 実行済みノートブックのパス
        
    Returns:
        ルートのリスト
    """
    import json
    import re
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # ノートブックのセルから解のルートを探す
    routes = []
    
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'outputs' in cell:
            for output in cell['outputs']:
                if 'text' in output:
                    text = ''.join(output['text'])
                    
                    # 日本語パターン: "ルート 1: [21, 22, 26, 23, 24, 27, 20, 9]"
                    route_matches = re.findall(r'ルート \d+: \[(.+?)\]', text)
                    
                    for match in route_matches:
                        # "21, 22, 26, 23, 24, 27, 20, 9" を [0, 21, 22, 26, 23, 24, 27, 20, 9, 0] に変換
                        # （デポット0を最初と最後に追加）
                        route_nodes = [int(x.strip()) for x in match.split(',')]
                        # デポット（0）を最初と最後に追加
                        full_route = [0] + route_nodes + [0]
                        routes.append(full_route)
                    
                    # 英語パターンもサポート（後方互換性のため）
                    english_matches = re.findall(r'Route for vehicle \d+: (.+)', text)
                    for match in english_matches:
                        # "0 -> 5 -> 3 -> 0" を [0, 5, 3, 0] に変換
                        route_nodes = [int(x.strip()) for x in match.split('->')]
                        routes.append(route_nodes)
    
    return routes


def create_solution_visualization(experiment_dir: str, experiment_id: str,
                                vrp_file: str, title_suffix: str = "") -> str:
    """
    実験ディレクトリから解を読み取って可視化を作成
    
    Args:
        experiment_dir: 実験ディレクトリのパス
        experiment_id: 実験ID（exp001など）
        vrp_file: VRPインスタンスファイルのパス
        title_suffix: タイトルに追加する文字列
        
    Returns:
        作成された可視化ファイルのパス
    """
    
    # ノートブックファイルからルートを抽出
    notebook_path = os.path.join(experiment_dir, f"{experiment_id}.ipynb")
    
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    routes = extract_routes_from_notebook_output(notebook_path)
    
    if not routes:
        raise ValueError("No routes found in notebook output")
    
    # 可視化を作成
    output_path = os.path.join(experiment_dir, f"{experiment_id}_solution_plot.png")
    title = f"VRP Solution - {experiment_id.upper()}{title_suffix}"
    
    return visualize_vrp_solution(vrp_file, routes, output_path, title)


def create_comparison_visualization(experiment_dir: str, experiment_id: str,
                                   vrp_file: str, sol_file: str, 
                                   our_cost: float, optimal_cost: float) -> str:
    """
    実験解と最適解を上下に並べて比較可視化を作成
    
    Args:
        experiment_dir: 実験ディレクトリのパス
        experiment_id: 実験ID（exp001など）
        vrp_file: VRPインスタンスファイルのパス
        sol_file: 最適解ファイルのパス
        our_cost: 我々が得た解のコスト
        optimal_cost: 最適解のコスト
        
    Returns:
        作成された比較可視化ファイルのパス
    """
    try:
        from .utils import read_solution
    except ImportError:
        from utils import read_solution
    
    # ノートブックファイルからルートを抽出
    notebook_path = os.path.join(experiment_dir, f"{experiment_id}.ipynb")
    
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    
    our_routes = extract_routes_from_notebook_output(notebook_path)
    
    if not our_routes:
        raise ValueError("No routes found in notebook output")
    
    # 最適解を読み込み
    optimal_routes, _ = read_solution(sol_file)
    
    # VRPデータを読み込み
    vrp_data = VRPDataReader(vrp_file)
    parsed_data = vrp_data.parse()
    
    # 上下比較プロットを作成（1:2の縦長、各プロットは正方形、余白最小）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16), 
                                   gridspec_kw={'hspace': 0})
    
    # 黄色を除外したソフトなカラーマップを使用
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'olive', 'navy', 'teal', 'maroon', 'indigo']
    our_colors = [colors[i % len(colors)] for i in range(len(our_routes))]
    optimal_colors = [colors[i % len(colors)] for i in range(len(optimal_routes))]
    
    depot_x, depot_y = vrp_data.node_coords[1]  # デポットは1番ノード
    
    # 上側：得られた解
    ax1.scatter(depot_x, depot_y, c='black', s=200, marker='s', label='デポット', zorder=5, 
                edgecolors='white', linewidth=2)
    
    # 顧客ノードをプロット（背景として）
    customer_x = []
    customer_y = []
    for vrp_node_id in sorted(vrp_data.node_coords.keys()):
        if vrp_node_id != 1:  # デポット以外
            coord = vrp_data.node_coords[vrp_node_id]
            customer_x.append(coord[0])
            customer_y.append(coord[1])
    
    ax1.scatter(customer_x, customer_y, c='lightgray', s=50, marker='o', 
                label='顧客', zorder=3, edgecolors='black', linewidth=0.5)
    
    for i, route in enumerate(our_routes):
        if len(route) <= 2:  # デポのみのルートはスキップ
            continue
            
        color = our_colors[i]
        
        # ルート座標を取得
        def get_coord(route_node_id):
            if route_node_id == 0:
                return vrp_data.node_coords[1]  # depot
            else:
                return vrp_data.node_coords[route_node_id]
        
        route_coords = [get_coord(node) for node in route]
        route_x = [coord[0] for coord in route_coords]
        route_y = [coord[1] for coord in route_coords]
        
        # ルートを線で結ぶ
        ax1.plot(route_x, route_y, color=color, linewidth=2, alpha=0.8,
                label=f'車両 {i + 1}', zorder=2)
        
        # ルート上の点を強調
        for coord in route_coords[1:-1]:  # デポットを除く
            ax1.scatter(coord[0], coord[1], color=color, s=80, marker='o', zorder=4,
                       edgecolors='white', linewidth=1)
    
    gap = ((our_cost - optimal_cost) / optimal_cost * 100) if optimal_cost > 0 else 0
    ax1.set_title(f'OR-tools解 (コスト: {our_cost}, ギャップ: {gap:.2f}%)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # 下側：最適解
    ax2.scatter(depot_x, depot_y, c='black', s=200, marker='s', label='Depot', zorder=5,
                edgecolors='white', linewidth=2)
    
    # 顧客ノードをプロット（背景として）
    ax2.scatter(customer_x, customer_y, c='lightgray', s=50, marker='o', 
                label='顧客', zorder=3, edgecolors='black', linewidth=0.5)
    
    for i, route in enumerate(optimal_routes):
        if not route:  # 空のルートはスキップ
            continue
            
        color = optimal_colors[i]
        
        # 最適解のルート座標を取得（添え字修正：.solの番号+1 = VRPファイルの番号）
        # .solファイル: 1-75はお客様、VRPファイル: 1はデポット、2-76はお客様
        corrected_route = [node + 1 for node in route]  # .solの番号をVRPファイルの番号に変換
        route_with_depot = [1] + corrected_route + [1]  # デポット(1)を始点・終点に追加
        route_coords = [vrp_data.node_coords[node] for node in route_with_depot]
        route_x = [coord[0] for coord in route_coords]
        route_y = [coord[1] for coord in route_coords]
        
        # ルートを線で結ぶ
        ax2.plot(route_x, route_y, color=color, linewidth=2, alpha=0.8, 
                label=f'車両 {i + 1}', zorder=2)
        
        # ルート上の点を強調（添え字修正済み）
        for node in corrected_route:  # デポットを除く顧客ノードのみ（既に+1済み）
            if node in vrp_data.node_coords:
                coord = vrp_data.node_coords[node]
                ax2.scatter(coord[0], coord[1], color=color, s=80, marker='o', zorder=4,
                           edgecolors='white', linewidth=1)
    
    ax2.set_title(f'最適解 (コスト: {optimal_cost})', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # レイアウト調整（余白は既にgridspec_kwで設定済み）
    plt.tight_layout(pad=0.5)
    
    # ファイルを保存
    output_path = os.path.join(experiment_dir, f"{experiment_id}_comparison_plot.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=80, bbox_inches='tight', facecolor='white')
    plt.close()  # メモリリークを防ぐ
    
    return output_path