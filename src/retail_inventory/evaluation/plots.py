import matplotlib.pyplot as plt
import seaborn as sns


def plot_cost_breakdown(inventory_table, output_path):
    cols = ["holding_cost", "stockout_cost", "ordering_cost"]
    df = inventory_table[["model"] + cols].copy()
    df = df.set_index("model")[cols]
    ax = df.plot(kind="bar", stacked=True, figsize=(9, 4), color=["#4C78A8", "#F58518", "#54A24B"])
    ax.set_ylabel("Cost")
    ax.set_xlabel("")
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=180)
    plt.close(ax.figure)


def plot_accuracy_cost_scatter(forecast_table, inventory_table, output_path):
    merged = forecast_table.merge(inventory_table, on="model", how="inner")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=merged, x="RMSE", y="total_cost", hue="model", s=90, ax=ax)
    for _, row in merged.iterrows():
        ax.text(row["RMSE"], row["total_cost"], row["model"], fontsize=8)
    ax.set_xlabel("Protection-period RMSE")
    ax.set_ylabel("Inventory total cost")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
