import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jijmodeling as jm
    from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter
    return OMMXPySCIPOptAdapter, jm, mo


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## フロー捕捉型配置問題の実装
    """)
    return


@app.cell
def _(jm):
    # パラメータ
    p = jm.Placeholder("p", dtype=jm.DataType.INTEGER, description="配置施設数")
    K = jm.Placeholder("K", dtype=jm.DataType.INTEGER, description="頂点数")
    Q = jm.Placeholder("Q", dtype=jm.DataType.INTEGER, description="フロー数")

    f = jm.Placeholder("f", dtype=jm.DataType.INTEGER, shape=(Q,), description="フローqの流量")
    a = jm.Placeholder("a", dtype=jm.DataType.INTEGER, shape=(Q, K), description="頂点kがフローqの経路上に含まれるとき1、そうでないとき0")

    # 決定変数
    x = jm.BinaryVar("x", shape=(K,), description="頂点kに施設を配置するとき1、そうでないとき0")
    y = jm.BinaryVar("y", shape=(Q, ), description="フローqが捕捉されるとき1、そうでないとき0")

    # 添字
    k = jm.Element("k", belong_to=(0, K))
    q = jm.Element("q", belong_to=(0, Q))

    # 定式化
    problem = jm.Problem("FCLP", sense=jm.ProblemSense.MAXIMIZE)

    ## 目的関数: 捕捉流量を最大化
    problem += jm.sum(q, f[q] * y[q])

    ## 制約式
    problem += jm.Constraint("施設配置数", jm.sum(k, x[k]) == p)
    problem += jm.Constraint("フロー捕捉", y[q] <= jm.sum(k, a[q, k] * x[k]), forall=q)

    ## 表示
    problem
    return (problem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ランダムなデータを入力
    """)
    return


@app.cell
def _():
    import random
    import rustworkx as rx
    from rustworkx.visualization import mpl_draw

    def generate_road_like_network(
        rows: int,
        cols: int,
        p_remove_node: float,
        seed: int
    ):
        """
        rows x cols の格子状ネットワークを作り、そこからランダムにノードを削除する

        戻り値:
            g: rx.PyGraph
            positions: dict[node_index] = (x, y)  可視化・デバッグ用
        """
        random.seed(seed)

        # 削除しないノードを決定し、連番のインデックスを割り当てる
        grid_to_node = {}  # (r, c) -> node_index
        positions = {}
        node_idx = 0
        for r in range(rows):
            for c in range(cols):
                if random.random() >= p_remove_node:
                    grid_to_node[(r, c)] = node_idx
                    positions[node_idx] = (c, r)
                    node_idx += 1

        # グラフを作成
        g = rx.PyGraph()
        g.add_nodes_from([None] * len(grid_to_node))

        # エッジを追加（両端が存在するもののみ）
        for (r, c), u in grid_to_node.items():
            if (r, c + 1) in grid_to_node:
                g.add_edge(u, grid_to_node[(r, c + 1)], {"length": 1})
            if (r + 1, c) in grid_to_node:
                g.add_edge(u, grid_to_node[(r + 1, c)], {"length": 1})

        return g, positions
    return generate_road_like_network, mpl_draw, random, rx


@app.cell
def _(generate_road_like_network, mpl_draw):
    g, positions = generate_road_like_network(rows=5, cols=5, p_remove_node=0.15, seed=8)
    mpl_draw(g, positions, with_labels=True, node_color="lightgray")
    return g, positions


@app.cell
def _(g, random, rx):
    def generate_random_flows(
        g: rx.PyGraph,
        num_flows: int,
        min_amount: int = 1,
        max_amount: int = 10,
    ):
        """
        グラフ g 上でランダムな OD フローを num_flows 個つくる。

        - source ≠ target のノードペアをランダムに選ぶ
        - amount は [min_amount, max_amount] の整数乱数
        """
        flows = []
        nodes = list(g.node_indices())
        all_shortest_paths = rx.graph_all_pairs_dijkstra_shortest_paths(g, lambda e: e["length"])

        for _ in range(num_flows):
            s, t = random.sample(nodes, 2)
            amount = random.randint(min_amount, max_amount)
            flows.append({"source": s, "target": t, "path": all_shortest_paths[s][t], "amount": amount})

        return flows

    flows = generate_random_flows(g, num_flows=100)
    # flows
    return flows, generate_random_flows


@app.cell
def _(flows, g, mpl_draw, positions):

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from collections import defaultdict

    # 日本語フォント設定
    plt.rcParams["font.family"] = "Hiragino Sans"

    def draw_flow_heatmap(graph, pos, flow_list):
        # 各ノードの通過フロー数をカウント
        node_counts = defaultdict(int)
        for fl in flow_list:
            for node in fl["path"]:
                node_counts[node] += fl["amount"]

        # 各エッジの通過フロー数をカウント
        edge_counts = defaultdict(int)
        for fl in flow_list:
            path = fl["path"]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_key = (min(u, v), max(u, v))
                edge_counts[edge_key] += fl["amount"]

        max_node_count = max(node_counts.values()) if node_counts else 1
        max_edge_count = max(edge_counts.values()) if edge_counts else 1
        cmap = plt.cm.Reds

        # ノードの色とサイズをリストで作成
        node_colors = []
        node_sizes = []
        for node in graph.node_indices():
            count = node_counts.get(node, 0)
            intensity = count / max_node_count if count > 0 else 0
            node_colors.append("lightgray")
            node_sizes.append(10 + 1000 * intensity)

        # エッジの色をリストで作成
        edge_colors = []
        for u, v in graph.edge_list():
            edge_key = (min(u, v), max(u, v))
            count = edge_counts.get(edge_key, 0)
            intensity = count / max_edge_count if count > 0 else 0
            edge_colors.append(cmap(0.3 + 0.7 * intensity) if count > 0 else "lightgray")

        fig, ax = plt.subplots(figsize=(10, 8))
        mpl_draw(
            graph, pos,
            ax=ax,
            with_labels=True,
            node_color=node_colors,
            node_size=node_sizes,
            edge_color=edge_colors,
            width=2,
            font_color="black",
        )

        # カラーバーを追加
        norm = mcolors.Normalize(vmin=0, vmax=max_edge_count)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.ax.yaxis.set_tick_params(color="black")
        cbar.ax.tick_params(labelcolor="black")

        return fig

    draw_flow_heatmap(g, positions, flows)
    return defaultdict, plt


@app.cell
def _(flows, g, mpl_draw, plt, positions):
    import matplotlib.cm as cm

    def draw_flows_detail(graph, pos, flow_list):
        fig, ax = plt.subplots(figsize=(10, 8))

        # ベースのネットワークを描画
        mpl_draw(
            graph, pos,
            ax=ax,
            with_labels=False,
            node_color="lightgray",
            edge_color="lightgray",
            width=2,
            font_color="black",
        )

        # 各フローを色分けして描画
        colors = cm.tab10.colors
        for idx, fl in enumerate(flow_list):
            color = colors[idx % len(colors)]
            path = fl["path"]
            # 経路のエッジを描画（少しずらして見やすく）
            offset = (idx - len(flow_list) / 2) * 0.05
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                # 最後のエッジは矢印で描画
                if i == len(path) - 2:
                    ax.annotate("", xy=(x1 + offset, y1 + offset), xytext=(x0 + offset, y0 + offset),
                                arrowprops=dict(arrowstyle="-|>", color=color, lw=2, mutation_scale=30),
                                zorder=4)
                else:
                    ax.plot([x0 + offset, x1 + offset], [y0 + offset, y1 + offset],
                            color=color, linewidth=2, alpha=0.8, zorder=4)
                # 凡例用（最初のエッジのみ）
                if i == 0:
                    ax.plot([], [], color=color, linewidth=2, label=f"Flow {idx} (amount={fl['amount']})")
            # sourceを強調
            sx, sy = pos[fl["source"]]
            ax.scatter(sx + offset, sy + offset, s=80, c=[color], marker="o", zorder=5)

        ax.axis("off")
        plt.tight_layout()
        return fig

    draw_flows_detail(g, positions, flows)
    return


@app.cell
def _(flows, g):
    # 頂点kがflows[q]のpath上に含まれるとき1、そうでないとき0をとるa_exampleをつくる
    a_example = []
    for qq, flow in enumerate(flows):
        a_example.append([])
        for kk in g.node_indices():
            if kk in flow["path"]:
                a_example[qq].append(1)
            else:
                a_example[qq].append(0)
    # a_example
    return (a_example,)


@app.cell
def _(a_example, flows, g):
    # ベースとなるデータセット（pは後で指定）
    base_dataset = {
        "K": len(g.nodes()),
        "Q": len(flows),
        "f": [f["amount"] for f in flows],
        "a": a_example
    }
    # base_dataset
    return (base_dataset,)


@app.cell
def _(OMMXPySCIPOptAdapter, base_dataset, jm, problem):
    # 施設数1〜10で解いてsolutionsに保存
    solutions = {}
    for p_val in range(1, 7):
        dataset = {**base_dataset, "p": p_val}
        instance = jm.Interpreter(dataset).eval_problem(problem)
        solutions[p_val] = OMMXPySCIPOptAdapter.solve(instance)
    solutions
    return (solutions,)


@app.cell
def _(flows):
    sum([f["amount"] for f in flows])
    return


@app.cell
def _(plt, solutions):
    # 施設数 vs 目的関数値の棒グラフ
    def draw_objective_bar_chart(sols):
        p_values = sorted(sols.keys())
        objectives = [sols[p].objective for p in p_values]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(p_values, objectives, color="steelblue")

        ax.set_xlabel("配置施設数 (p)", fontsize=12, color="black")
        ax.set_ylabel("捕捉できた流量", fontsize=12, color="black")
        ax.set_xticks(p_values)
        ax.tick_params(colors="black")

        # 各バーの上に値を表示
        for bar, obj in zip(bars, objectives):
            ax.annotate(f"{obj:.0f}", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2),
                        ha="center", va="bottom", fontsize=10, color="black")

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # 枠を追加
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

        plt.tight_layout()
        return fig

    draw_objective_bar_chart(solutions)
    return


@app.cell
def _(defaultdict, flows, g, mpl_draw, plt, positions, solutions):
    # 施設数1〜10の施設配置を一覧表示
    def draw_all_solutions(graph, pos, flow_list, sols):
        # 各ノードの通過フロー数をカウント（共通）
        node_counts = defaultdict(int)
        for fl in flow_list:
            for node in fl["path"]:
                node_counts[node] += fl["amount"]

        # 各エッジの通過フロー数をカウント（共通）
        edge_counts = defaultdict(int)
        for fl in flow_list:
            path = fl["path"]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_key = (min(u, v), max(u, v))
                edge_counts[edge_key] += fl["amount"]

        max_node_count = max(node_counts.values()) if node_counts else 1
        max_edge_count = max(edge_counts.values()) if edge_counts else 1
        cmap = plt.cm.Reds

        # ノードサイズとエッジ色（共通）
        node_sizes = []
        for node in graph.node_indices():
            count = node_counts.get(node, 0)
            intensity = count / max_node_count if count > 0 else 0
            node_sizes.append(10 + 1000 * intensity)

        edge_colors = []
        for u, v in graph.edge_list():
            edge_key = (min(u, v), max(u, v))
            count = edge_counts.get(edge_key, 0)
            intensity = count / max_edge_count if count > 0 else 0
            edge_colors.append(cmap(0.3 + 0.7 * intensity) if count > 0 else "lightgray")

        # 2行5列のサブプロット
        fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        axes = axes.flatten()

        for idx, p_val in enumerate(range(1, 7)):
            ax = axes[idx]
            sol = sols[p_val]

            # x==1のノードを取得
            df = sol.decision_variables_df
            x_nodes = set(df[(df["name"].str.startswith("x")) & (df["value"] == 1)]["subscripts"].apply(lambda s: s[0]).tolist())

            # ベースネットワーク描画
            mpl_draw(
                graph, pos,
                ax=ax,
                with_labels=True,
                node_color="lightgray",
                node_size=node_sizes,
                edge_color=edge_colors,
                width=2,
            )

            # x==1のノードを黒丸で上書き描画
            for node in x_nodes:
                nx, ny = pos[node]
                count = node_counts.get(node, 0)
                intensity = count / max_node_count if count > 0 else 0
                size = 10 + 1000 * intensity
                ax.scatter(nx, ny, s=size, c="black", marker="o", zorder=10)
                ax.annotate(str(node), (nx, ny), ha="center", va="center", fontsize=8, color="white", fontweight="bold", zorder=11)

            # 図の下にタイトルをテキストで配置
            ax.text(0.5, 0.05, f"$p={p_val}, obj={sol.objective:.0f}$",
                    transform=ax.transAxes, ha="center", va="top", fontsize=16, color="black")

        fig.patch.set_facecolor("white")
        plt.tight_layout()
        return fig

    draw_all_solutions(g, positions, flows, solutions)
    return


@app.cell
def _(defaultdict, flows, g, mpl_draw, plt, positions, solutions):
    # p=2のネットワーク図
    def draw_single_solution(graph, pos, flow_list, sol, p_val):
        # 各ノードの通過フロー数をカウント
        node_counts = defaultdict(int)
        for fl in flow_list:
            for node in fl["path"]:
                node_counts[node] += fl["amount"]

        # 各エッジの通過フロー数をカウント
        edge_counts = defaultdict(int)
        for fl in flow_list:
            path = fl["path"]
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                edge_key = (min(u, v), max(u, v))
                edge_counts[edge_key] += fl["amount"]

        max_node_count = max(node_counts.values()) if node_counts else 1
        max_edge_count = max(edge_counts.values()) if edge_counts else 1
        cmap = plt.cm.Reds

        # ノードサイズとエッジ色
        node_sizes = []
        for node in graph.node_indices():
            count = node_counts.get(node, 0)
            intensity = count / max_node_count if count > 0 else 0
            node_sizes.append(10 + 1000 * intensity)

        edge_colors = []
        for u, v in graph.edge_list():
            edge_key = (min(u, v), max(u, v))
            count = edge_counts.get(edge_key, 0)
            intensity = count / max_edge_count if count > 0 else 0
            edge_colors.append(cmap(0.3 + 0.7 * intensity) if count > 0 else "lightgray")

        # x==1のノードを取得
        df = sol.decision_variables_df
        x_nodes = set(df[(df["name"].str.startswith("x")) & (df["value"] == 1)]["subscripts"].apply(lambda s: s[0]).tolist())

        fig, ax = plt.subplots(figsize=(10, 8))
        mpl_draw(
            graph, pos,
            ax=ax,
            with_labels=True,
            node_color="lightgray",
            node_size=node_sizes,
            edge_color=edge_colors,
            width=2,
            font_color="black",
        )

        # x==1のノードを黒丸で上書き描画
        for node in x_nodes:
            nx, ny = pos[node]
            count = node_counts.get(node, 0)
            intensity = count / max_node_count if count > 0 else 0
            size = 10 + 1000 * intensity
            ax.scatter(nx, ny, s=size, c="black", marker="o", zorder=10)
            ax.annotate(str(node), (nx, ny), ha="center", va="center", fontsize=10, color="white", fontweight="bold", zorder=11)

        ax.text(0.5, 0.02, f"$p={p_val}, obj={sol.objective:.0f}$",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=14, color="black")

        fig.patch.set_facecolor("white")
        plt.tight_layout()
        return fig

    draw_single_solution(g, positions, flows, solutions[2], p_val=2)
    return


@app.cell
def _(OMMXPySCIPOptAdapter, g, generate_random_flows, jm, problem):
    import time

    # フロー数を変えたときの計算時間を計測
    def benchmark_by_flow_count(graph, p_val, flow_counts):
        results = {}

        for num_flows in flow_counts:
            # フローを生成
            test_flows = generate_random_flows(graph, num_flows=num_flows)

            # a_exampleを作成
            a_test = []
            for fl in test_flows:
                row = [1 if k in fl["path"] else 0 for k in graph.node_indices()]
                a_test.append(row)

            # データセット作成
            dataset = {
                "p": p_val,
                "K": len(graph.nodes()),
                "Q": len(test_flows),
                "f": [f["amount"] for f in test_flows],
                "a": a_test
            }

            # 最適化を実行して時間計測
            instance = jm.Interpreter(dataset).eval_problem(problem)
            start_time = time.time()
            sol = OMMXPySCIPOptAdapter.solve(instance)
            elapsed_time = time.time() - start_time

            results[num_flows] = {
                "time": elapsed_time,
                "objective": sol.objective
            }
            print(f"フロー数: {num_flows}, 時間: {elapsed_time:.2f}秒, 目的関数値: {sol.objective:.0f}")

        return results

    flow_counts = [100, 1000, 10000, 50000, 100000, 500000]
    benchmark_results = benchmark_by_flow_count(g, p_val=5, flow_counts=flow_counts)
    benchmark_results
    return (benchmark_results,)


@app.cell
def _(benchmark_results, plt):
    # フロー数 vs 計算時間の棒グラフ
    def draw_benchmark_chart(results):
        flow_counts = sorted(results.keys())
        times = [results[n]["time"] for n in flow_counts]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(range(len(flow_counts)), times, color="steelblue")

        ax.set_xlabel("フロー数", fontsize=12, color="black")
        ax.set_ylabel("計算時間 (秒)", fontsize=12, color="black")
        ax.set_xticks(range(len(flow_counts)))
        ax.set_xticklabels([f"{n:,}" for n in flow_counts])
        ax.tick_params(colors="black")

        # 各バーの上に値を表示
        for bar, t in zip(bars, times):
            ax.annotate(f"{t:.2f}s", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha="center", va="bottom", fontsize=10, color="black")

        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        # 枠を追加
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1)

        plt.tight_layout()
        return fig

    draw_benchmark_chart(benchmark_results)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
