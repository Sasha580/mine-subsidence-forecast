import matplotlib.pyplot as plt


def plot_profiles(df, title=None, caption=None, figsize=(15, 10)):
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

    # TODO
    # if caption:
    #     fig.text(0.5, 0.08, caption, ha='center', va="top", wrap=True, fontsize=12)

    ax.set_ylabel("Subsidence (mm)")
    ax.set_xlabel("Coordinate along the length of the mine")
    ax.legend(fontsize=20, bbox_to_anchor=(1.05, 1))
    fig.tight_layout(rect=(0, 0.05, 1, 1))  # Reserve space for caption
    # fig.subplots_adjust(right=0.75)
    return fig


def plot_training(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(train_losses, label="Training Loss")
    ax.plot(val_losses, label="Validation Loss")
    ax.legend()
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    return fig