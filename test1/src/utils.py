from os.path import join
import os
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.style.use('ggplot')


def save_model(epochs, model, optimizer, criterion, pretrained, filename: list[str]=["model_pretrained.pth"]):
    """
    Function to save the trained model to disk.
    """
    os.makedirs(join("..", "outputs", *filename[:-1]), exist_ok=True)
    torch.save(
        {
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        },
        join("..", "outputs", *filename)
    )


def save_plots(train_acc, valid_acc, train_loss, valid_loss, pretrained, file_path: list[str]):
    """
    Function to save the loss and accuracy plots to disk.
    """

    base_file_path = join("..", "outputs", *file_path)

    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-',
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-',
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(join(base_file_path, "accuracy_graph.png"))

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss, color='orange', linestyle='-',
        label='train loss'
    )
    plt.plot(
        valid_loss, color='red', linestyle='-',
        label='validataion loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(join(base_file_path, "loss_graph.png"))


def draw_loss_matrix(matrix: dict[str, dict[str, int]]):
    max_key_len: int = len(max(matrix, key=len))
    matrix: dict[str, list[str]] = {k: ["" + str(v.get(i, 0)).center(3) + "" for i in matrix] for k, v in matrix.items()}
    upper_titles = [k.rjust(max_key_len) for k in list(matrix.keys())]
    matrix: dict[str, list[str]] = {k.ljust(max_key_len + 1) + "|": v for k, v in matrix.items()}

    print(upper_titles)
    len_data = 0
    for i in range(max_key_len ):
        data = " " * (max_key_len + 1) + "|" + "".join(["" + str(k[i]).center(3) + "" for k in upper_titles])
        print(data)
        len_data = len(data)
    print("_" * len_data)
    for k, v in matrix.items():
        data = k+"".join(v)
        print(data)
        print("-" * len(data))

if __name__ == "__main__":
    draw_loss_matrix(
        {'elephant': {'elephant': 11, 'bison': 1}, 'white_horses': {'white_horses': 8}, 'cow': {'cow': 10, 'bison': 1},
         'sheep': {'sheep': 9, 'bison': 1, 'winter': 1}, 'hippopotamus': {'hippopotamus': 10}, 'bison': {'bison': 8},
         'donkey': {'donkey': 7, 'white_horses': 1}, 'turtle': {'turtle': 12}, 'antelope': {'antelope': 8},
         'zebra': {'zebra': 13}, 'rhinoceros': {'rhinoceros': 11}, 'deer': {'deer': 10}, 'winter': {'winter': 10},
         'kangaroo': {'kangaroo': 8}, 'horse': {'horse': 7, 'donkey': 1, 'white_horses': 1}}

    )