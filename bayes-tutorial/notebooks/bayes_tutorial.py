import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    ## ベイズ推論（ベルヌーイ版）

    - [1, 0, 0, 1, 0] という観測値から推論する
    """)
    return


@app.cell
def _(np):
    import pymc as pm
    X = np.array([1, 0, 0, 1, 0])
    model1 = pm.Model()
    with model1:
        p = pm.Uniform("p", lower=0.0, upper=1.0)
        X_obs = pm.Bernoulli("X_obs", p=p, observed=X)
    return model1, pm


@app.cell
def _(model1):
    model1
    return


@app.cell
def _(model1, pm):
    with model1:
        idata1_1 = pm.sample(
            chains=3,  # 乱数系列の数
            tune=2000,  # 捨てるサンプル数
            draws=2000,  # 取得するサンプル数
            random_seed=123
        )
    return (idata1_1,)


@app.cell
def _(idata1_1):
    idata1_1.observed_data["X_obs"].values
    return


@app.cell
def _(idata1_1):
    idata1_1.posterior["p"].values
    return


@app.cell
def _(idata1_1, plt):
    import arviz as az

    az.plot_trace(idata1_1, compact=False)
    plt.tight_layout()
    plt.show()
    return (az,)


@app.cell
def _(az, idata1_1):
    import matplotlib_fontja
    ax = az.plot_posterior(idata1_1)
    ax.set_xlim(0, 1)
    ax.set_title("ベイズ推論結果")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ベイズ推論（二項分布版）

    - 「5回中2回当たり」という情報から推論する
    """)
    return


@app.cell
def _(pm):
    model2 = pm.Model()
    with model2:
        p2 = pm.Uniform("p", lower=0.0, upper=1.0)
        X_obs2 = pm.Binomial("X_obs", p=p2, n=5, observed=2)
    return (model2,)


@app.cell
def _(model2):
    model2
    return


@app.cell
def _(model2, pm):
    with model2:
        idata2_1 = pm.sample(
            chains=3,  # 乱数系列の数
            tune=2000,  # 捨てるサンプル数
            draws=2000,  # 取得するサンプル数
            random_seed=123
        )
    return (idata2_1,)


@app.cell
def _(az, idata2_1, plt):
    az.plot_trace(idata2_1, compact=False)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(az, idata2_1):
    ax2 = az.plot_posterior(idata2_1)
    ax2.set_xlim(0, 1)
    ax2.set_title("ベイズ推論結果")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ベイズ推論（試行回数を増やす）

    - 「50回中20回当たり」という情報から推論する
    """)
    return


@app.cell
def _(pm):
    model3 = pm.Model()
    with model3:
        p3 = pm.Uniform("p", lower=0.0, upper=1.0)
        X_obs3 = pm.Binomial("X_obs", p=p3, n=50, observed=20)
    return (model3,)


@app.cell
def _(model3):
    model3
    return


@app.cell
def _(model3, pm):
    with model3:
        idata3 = pm.sample(
            chains=3,  # 乱数系列の数
            tune=2000,  # 捨てるサンプル数
            draws=2000,  # 取得するサンプル数
            random_seed=123
        )
    return (idata3,)


@app.cell
def _(az, idata3, plt):
    az.plot_trace(idata3, compact=False)
    plt.tight_layout()
    plt.show()
    return


@app.cell
def _(az, idata3):
    ax3 = az.plot_posterior(idata3)
    ax3.set_xlim(0, 1)
    ax3.set_title("ベイズ推論結果")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## ベイズ推論（事前分布を変える）

    - 0 <= p <= 1 ではなく 0.1 <= p <= 0.9 にする
    """)
    return


@app.cell
def _(az, pm):
    model4 = pm.Model()
    with model4:
        p4 = pm.Uniform("p", lower=0.1, upper=0.9)
        X_obs4 = pm.Binomial("X_obs", p=p4, n=5, observed=2)
        idata4 = pm.sample(
            chains=3,  # 乱数系列の数
            tune=2000,  # 捨てるサンプル数
            draws=2000,  # 取得するサンプル数
            random_seed=123
        )
    ax4 = az.plot_posterior(idata4)
    ax4.set_xlim(0, 1)
    ax4.set_title("ベイズ推論結果")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
