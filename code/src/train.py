import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import time
from tqdm.auto import tqdm
from src.model import build_model
from src.datasets import get_datasets, get_data_loaders
from src.utils import save_model, save_plots, draw_loss_matrix
import os
from datetime import datetime


# Training function.
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    res_matrix: dict[int, dict[int, int]] = dict()
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)

        loc_res_matrix = {
            int(i[2]): int(i[1])
            for i in [
                max([(j, ind, labels.data[ind1]) for ind, j in enumerate(i)], key=lambda i1: i1[0])
                for ind1, i in enumerate(outputs.data)
            ]
        }

        for k, v in loc_res_matrix.items():
            res_matrix[k] = res_matrix.get(k, dict())
            res_matrix[k][v] = res_matrix[k].get(v, 0) + 1

        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation
        loss.backward()
        # Update the weights.
        optimizer.step()


    code_to_image_class_name: dict[int, str] = {v: k for k, v in trainloader.dataset.dataset.class_to_idx.items()}
    res_matrix: dict[str, dict[str, int]] = {
        code_to_image_class_name[k1]: {code_to_image_class_name[k2]: v2 for k2, v2 in v1.items()} for k1, v1 in
        res_matrix.items()}

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc, res_matrix


# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    res_matrix: dict[int, dict[int, int]] = dict()

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)

            loc_res_matrix = {
                int(i[2]): int(i[1])
                for i in [
                    max([(j, ind, labels.data[ind1]) for ind, j in enumerate(i)], key=lambda i1: i1[0])
                    for ind1, i in enumerate(outputs.data)
                ]
            }

            for k, v in loc_res_matrix.items():
                res_matrix[k] = res_matrix.get(k, dict())
                res_matrix[k][v] = res_matrix[k].get(v, 0) + 1

            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    print()
    code_to_image_class_name: dict[int, str] = {v: k for k, v in testloader.dataset.dataset.class_to_idx.items()}
    res_matrix: dict[str, dict[str, int]] = {code_to_image_class_name[k1]: {code_to_image_class_name[k2]: v2 for k2, v2 in v1.items()} for k1, v1 in res_matrix.items()}

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc, res_matrix


if __name__ == '__main__':
    new_model_directory = input("\nВведите папку для сохранения модели:\n")

    # construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-e', '--epochs', type=int, default=int(input("\nВведите количество эпох: \n")),
        help='Number of epochs to train our network for'
    )
    parser.add_argument(
        '-pt', '--pretrained', action='store_true',
        default=int(input("\nСледует ли использовать предварительно подготовленные веса (1) или нет (0): \n")) == 1,
        help='Whether to use pretrained weights or not'
    )
    model_filename = input("\nВведите путь и имя файла, в котором хранятся веса модели (или пустую стоку) \n")
    if bool(model_filename) and os.path.isfile(model_filename := os.path.join("../../test1", "outputs", model_filename)):
        print(f"Будет использоваться файл весов: {model_filename}")
    else:
        model_filename = None
        print("Пользовательские веса не будут загружены")

    parser.add_argument(
        '-lr', '--learning-rate', type=float,
        dest='learning_rate', default=float(input("\nВведите скорость обучения (по умолчанию 0.0001): \n")),
        help='Learning rate for training the model'
    )
    args = vars(parser.parse_args())

    # Load the training and validation datasets.
    dataset_train, dataset_valid, dataset_classes = get_datasets(args['pretrained'])
    print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    print(f"[INFO]: Class names: {dataset_classes}\n")
    # Load the training and validation data loaders.
    train_loader, valid_loader = get_data_loaders(dataset_train, dataset_valid)
    # Learning_parameters.
    lr = args['learning_rate']
    epochs = args['epochs']
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")
    print(f"Learning rate: {lr}")
    print(f"Epochs to train for: {epochs}\n")
    model = build_model(
        pretrained=args['pretrained'],
        fine_tune=True,
        num_classes=len(dataset_classes),
        filename=model_filename,
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()
    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    best_acc = -1
    best_loss = float("inf")
    best_precision = -1
    last_division_epoch = 0
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc, train_matrix = train(model, train_loader,
                                                  optimizer, criterion, device)
        valid_epoch_loss, valid_epoch_acc, validate_matrix = validate(model, valid_loader,
                                                     criterion, device)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)

        TP_training = train_matrix.get('white_horses', dict()).get('white_horses', 0)
        FP_training = sum([v.get('white_horses', 0) for k, v in train_matrix.items() if k != "white_horses"])

        TP_validation = validate_matrix.get('white_horses', dict()).get('white_horses', 0)
        FP_validation = sum([v.get('white_horses', 0) for k, v in validate_matrix.items() if k != "white_horses"])

        if best_acc < valid_epoch_acc or (best_acc == valid_epoch_acc and best_loss > valid_epoch_loss):
            save_model(epoch, model, optimizer, criterion, args['pretrained'], [new_model_directory, "best_acc.pth"])

        current_precision = TP_validation / (TP_validation + FP_validation)
        if current_precision > best_precision or (current_precision == best_precision and best_acc < valid_epoch_acc):
            save_model(epoch, model, optimizer, criterion, args['pretrained'], [new_model_directory, "best_precision.pth"])

        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print(f"White horse precision metric:\n"
              f"\tTraining: {TP_training / (TP_training + FP_training):.3f}, \n"
              f"\tValidation acc: {TP_validation / (TP_validation + FP_validation):.3f}")

        if epoch % 5 == 0:
            print("train_matrix:")
            draw_loss_matrix(train_matrix)
            print("\nvalidate_matrix:")
            draw_loss_matrix(validate_matrix)

        best_acc = max(best_acc, valid_epoch_acc)
        best_loss = min(best_loss, valid_epoch_loss)


        if len(valid_acc) > 15 and epoch - last_division_epoch > 10:
            last_acc = valid_acc[-10:] + [-1]
            count = 0
            last_i = last_acc[0]
            posl = 0
            for ind, i in enumerate(last_acc[1:], 1):
                if i < last_i:
                    posl += 1
                else:
                    posl = 0
                    count += 1
                last_i = i
            if count >= 3 and max(last_acc) - min([i for i in last_acc if i >= 0]) < 10:
                lr /= 10
                last_division_epoch = epoch
                print("\n" + ("=" * 20) + f"! lr = {lr} !" + ("=" * 20), "\n")
                optimizer = optim.Adam(model.parameters(), lr=lr)

        print(datetime.utcnow().strftime("%H:%M:%S -%d-%b-%Y По Гринвичу").rjust(50, "-"))

        save_model(epoch, model, optimizer, criterion, args['pretrained'], [new_model_directory, "latest_epoch.pth"])
        time.sleep(5)

    # Save the trained model weights.
    save_model(epochs, model, optimizer, criterion, args['pretrained'], [new_model_directory, "latest_epoch.pth"])
    # Save the loss and accuracy plots.
    save_plots(train_acc, valid_acc, train_loss, valid_loss, args['pretrained'], [new_model_directory])
    print('TRAINING COMPLETE')
