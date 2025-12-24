import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pymc as pm
    import arviz as az
    return az, pm


@app.cell
def _(pm):
    model = pm.Model()

    with model:
        x = pm.Binomial("x", p=0.5, n=5)
        priopr_samples = pm.sample_prior_predictive(random_seed=42)
    priopr_samples
    return (priopr_samples,)


@app.cell
def _(az, priopr_samples):
    summary = az.summary(priopr_samples, kind="stats")
    return (summary,)


@app.cell
def _(summary):
    summary
    return


@app.cell
def _(az, priopr_samples):
    ax = az.plot_dist(priopr_samples["prior"]["x"].values)
    ax.set_title("sample")
    return


@app.cell
def _(az, pm):
    # ベルヌーイ分布
    p = 0.5

    model1 = pm.Model()
    with model1:
        x1 = pm.Bernoulli("x", p=p)
        priopr_samples1 = pm.sample_prior_predictive(random_seed=123)
    x_samples1 = priopr_samples1["prior"]["x"].values

    summary1 = az.summary(priopr_samples1, kind="stats")
    summary1
    return p, x_samples1


@app.cell
def _(az, p, x_samples1):
    import matplotlib_fontja
    ax1 = az.plot_dist(x_samples1)
    ax1.set_title(f"ベルヌーイ分布 p={p}")
    return


@app.cell
def _(az, pm):
    # 二項分布
    model2 = pm.Model()
    with model2:
        x2 = pm.Binomial("x", p=0.5, n=50)
        priopr_samples2 = pm.sample_prior_predictive(random_seed=123)
    x_samples2 = priopr_samples2["prior"]["x"].values

    summary2 = az.summary(priopr_samples2, kind="stats")

    ax2 = az.plot_dist(x_samples2)
    ax2.set_title(f"二項分布")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
