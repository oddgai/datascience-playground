import marimo

__generated_with = "0.18.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import jijmodeling as jm
    return jm, mo


@app.cell
def _(mo):
    mo.md(r"""
    ## ナップサック問題
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 定式化
    """)
    return


@app.cell
def _(jm):
    # パラメータ
    N = jm.Placeholder("N", dtype=jm.DataType.INTEGER)
    v = jm.Placeholder("v", dtype=jm.DataType.INTEGER, shape=(N,))  # アイテムの価値
    w = jm.Placeholder("w", dtype=jm.DataType.INTEGER, shape=(N,))  # 重さ
    W = jm.Placeholder("W", dtype=jm.DataType.INTEGER)  # 耐荷重

    # 決定変数
    x = jm.BinaryVar("x", shape=(N,))  # アイテムiを入れる場合は1、それ以外は0をとる変数
    i = jm.Element("i", belong_to=(0, N))  # アイテムの添字

    # 定式化
    problem = jm.Problem("napsack", sense=jm.ProblemSense.MAXIMIZE)

    ## 目的関数: ナップサック内のアイテム価値を最大化
    problem += jm.sum(i, v[i] * x[i])

    ## 制約式: ナップサックの耐荷重を超えない
    problem += jm.Constraint("重量制限", jm.sum(i, w[i] * x[i]) <= W)


    ## 表示
    problem
    return (problem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### `Interpreter`にデータを登録し、数理モデルをインスタンスに変換
    """)
    return


@app.cell
def _(jm, problem):
    # データを登録
    instance_data = {
        "N": 5,
        "v": [10, 13, 18, 31, 7, 15],  # アイテムの価値のデータ
        "w": [11, 15, 20, 35, 10, 33], # アイテムの重さのデータ
        "W": 47,                       # ナップサックの耐荷重のデータ
    }
    interpreter = jm.Interpreter(instance_data)

    # インスタンスに変換
    instance = interpreter.eval_problem(problem)
    return (instance,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 最適化問題を解く
    """)
    return


@app.cell
def _(instance):
    from ommx_highs_adapter import OMMXHighsAdapter
    from ommx_pyscipopt_adapter import OMMXPySCIPOptAdapter

    # 利用するAdapterの一覧
    adapters = {
        "highs": OMMXHighsAdapter,
        "scip": OMMXPySCIPOptAdapter,
    }

    # 各Adapterを介して問題を解く
    solutions = {
        name: adapter.solve(instance) for name, adapter in adapters.items()
    }
    return OMMXHighsAdapter, OMMXPySCIPOptAdapter, solutions


@app.cell
def _(solutions):
    solutions["scip"].objective
    return


@app.cell
def _(solutions):
    solutions["scip"].decision_variables_df
    return


@app.cell
def _(solutions):
    solutions["scip"].constraints_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ランダムなデータを入力する
    """)
    return


@app.cell
def _(problem):
    random_dataset = problem.generate_random_dataset(
        options={
            "N": {"value": 10**5},
            "v": {"value": range(1, 10)},
            "w": {"value": range(1, 10)},
            "W": {"value": 10**6},
        },
        seed=123
    )
    random_dataset
    return (random_dataset,)


@app.cell
def _(jm, problem, random_dataset):
    random_instance = jm.Interpreter(random_dataset).eval_problem(problem)
    return (random_instance,)


@app.cell
def _(OMMXHighsAdapter, random_instance):
    solution = OMMXHighsAdapter.solve(random_instance)
    return (solution,)


@app.cell
def _(OMMXPySCIPOptAdapter, random_instance):
    solution_scip = OMMXPySCIPOptAdapter.solve(random_instance)
    return (solution_scip,)


@app.cell
def _(solution):
    solution.objective
    return


@app.cell
def _(solution_scip):
    solution_scip.objective
    return


@app.cell
def _(solution):
    solution.decision_variables_df
    return


@app.cell
def _(solution):
    solution.constraints_df
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
