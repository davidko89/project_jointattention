#%%
import torch
import torch.utils
import matplotlib.pyplot as plt

from custom_model import CustomNet
from pathlib import Path


PROJECT_PATH = Path(__file__).parents[1]
DATA_PATH = Path(PROJECT_PATH, "data")
CHECKPOINT_PATH = Path(PROJECT_PATH, "checkpoint/lrcn_atten_rja_high_weight_10.pt")


def show_heatmaps(
    matrices, xlabel, ylabel, titles=None, figsize=(2.5, 2.5), cmap="Reds"
):
    """show heatmaps of matrices"""
    num_rows, num_cols = matrices.shape[0], matrices.shape[1]
    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=figsize, sharex=True, sharey=True, squeeze=False
    )

    for i, (row_axes, row_matrices) in enumerate(zip(axes, matrices)):
        for j, (ax, matrix) in enumerate(zip(row_axes, row_matrices)):
            pcm = ax.imshow(matrix.detach().numpy(), cmap=cmap)
            if i == num_rows - 1:
                ax.set_xlabel(xlabel)
            if j == 0:
                ax.set_ylabel(ylabel)
            if titles:
                ax.set_title(titles[j])

        fig.colorbar(pcm, ax=axes, shrink=0.6)


def main():
    model = CustomNet(
        input_size=25088,
        seq_len=150,
        num_hiddens=128,
        num_layers=2,
        dropout=0.5,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))

    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    attention_weights = model.attention.linear.weight

    show_heatmaps(
        attention_weights,
        xlabel="Sorted training inputs",
        ylabel="Sorted testing inputs",
    )
    
#%%
if __name__ == "__main__":
    main()
