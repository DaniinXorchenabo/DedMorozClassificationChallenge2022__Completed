import argparse
import torch.nn as nn
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from src.datasets import get_valid_transform, IMAGE_SIZE, ROOT_DIR, BATCH_SIZE, NUM_WORKERS
from src.train import validate
from src.utils import draw_loss_matrix

from src.model import build_model


if __name__ == '__main__':

    # construct the argument parser
    parser = argparse.ArgumentParser()
    model_filename = "../../test1/outputs/test1_b3/best_acc.pth"

    # Load the training and validation datasets.
    # dataset_train, dataset_valid, dataset_classes = get_datasets(True, valid_split=1)
    dataset_test = datasets.ImageFolder(
        ROOT_DIR,
        transform=(get_valid_transform(IMAGE_SIZE, True))
    )

    indices = torch.randperm(len(dataset_test)).tolist()
    # Training and validation sets.
    dataset_valid = Subset(dataset_test, indices)

    # print(f"[INFO]: Number of training images: {len(dataset_train)}")
    print(f"[INFO]: Number of validation images: {len(dataset_valid)}")
    # Load the training and validation data loaders.
    valid_loader = DataLoader(
        dataset_valid, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS
    )
    # Learning_parameters.
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computation device: {device}")

    model = build_model(
        pretrained=True,
        fine_tune=False,
        num_classes=len(dataset_test.classes),
        filename=model_filename,
    ).to(device)

    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    criterion = nn.CrossEntropyLoss()

    valid_epoch_loss, valid_epoch_acc, validate_matrix = validate(model, valid_loader, criterion, device)
    print(validate_matrix)
    draw_loss_matrix(validate_matrix)

    print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")


    TP_validation = validate_matrix.get('white_horses', dict()).get('white_horses', 0)
    FP_validation = sum([v.get('white_horses', 0) for k, v in validate_matrix.items() if k != "white_horses"])

    current_precision = TP_validation / (TP_validation + FP_validation)

    print(f"White horse precision metric:\n"
          f"\tValidation acc: {TP_validation / (TP_validation + FP_validation):.3f}")
