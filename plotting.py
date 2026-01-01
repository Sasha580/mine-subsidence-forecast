import matplotlib.pyplot as plt


def plot_profiles(df, title=None, figsize=(15, 10)):
    fig, ax = plt.subplots(figsize=figsize)

    x = df.iloc[:, 0]  # Use the first column as x-axis
    y_columns = df.columns[1:]  # All other columns are y-values

    for col in y_columns:
        if "Pred" in col:
            ax.plot(x, df[col], label=col, linestyle="--", alpha=0.5, linewidth=3, color="red")
        else:
            ax.plot(x, df[col], label=col)

    if title:
        ax.set_title(title)

    ax.set_ylabel("Value")
    ax.legend(fontsize=20, bbox_to_anchor=(1.05, 1))
    fig.subplots_adjust(right=0.75)
    return fig