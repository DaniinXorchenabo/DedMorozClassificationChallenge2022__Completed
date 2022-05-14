from os.path import dirname, join, split, isfile, isdir
import os

import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import dotenv

from src.datasets import get_valid_transform, IMAGE_SIZE, ROOT_DIR, BATCH_SIZE, NUM_WORKERS
from src.train import validate
from src.utils import draw_loss_matrix

from src.model import build_model


import requests

def download_file_from_google_drive(id, destination):
    # https://drive.google.com/file/d/1KsotujU1g2Yi2ivbUJuwo5fPWq_pBwzD/view?usp=sharing
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == '__main__':

    path = dirname(__file__)
    while "code" in (path := split(path)[0]):
        pass
    HOME_DIR = path


    # construct the argument parser
    parser = argparse.ArgumentParser()
    model_filename = join(HOME_DIR, "weights")
    os.makedirs(model_filename, exist_ok=True)
    model_filename = join(model_filename, "final_network.pth")
    if isfile(model_filename) is False:
        raise FileNotFoundError(f"Отсутствует файл с весами нейронной сети ({model_filename})")

    if isfile(join(HOME_DIR, '.env')):
        dotenv.load_dotenv(join(HOME_DIR, '.env'))

    ROOT_DIR = os.environ.get("DATA_PATH", None)
    if ROOT_DIR is None:
        raise EnvironmentError("Не найдена переменная окружающей среды DATA_PATH,"
                               " которая указывает на расположение тестируемых данных")

    if isdir(ROOT_DIR) is False:
        raise FileNotFoundError(f"По пути {ROOT_DIR} не найдено данных")

    # os.makedirs(join(ROOT_DIR, "winter"), exist_ok=True)

    # Load the training and validation datasets.
    # dataset_train, dataset_valid, dataset_classes = get_datasets(True, valid_split=1)
    dataset_test = datasets.ImageFolder(
        ROOT_DIR,
        transform=(get_valid_transform(IMAGE_SIZE, True))
    )
    dataset_test.classes = sorted(list(set(dataset_test.classes + ["winter"])))
    dataset_test.imgs = [(path, (cls + 1 if cls >= 13 else cls)) for [path, cls] in dataset_test.imgs]
    dataset_test.samples = [(path, (cls + 1 if cls >= 13 else cls)) for [path, cls] in dataset_test.samples]
    dataset_test.class_to_idx["zebra"] = 14
    dataset_test.class_to_idx["winter"] = 13
    dataset_test.targets = [(i + 1 if i >= 13 else i) for i in dataset_test.targets]

    indices = torch.randperm(len(dataset_test)).tolist()

    dataset_size = len(dataset_test)
    valid_size = int(0.1 * dataset_size)

    # Training and validation sets.
    dataset_valid = Subset(dataset_test, indices)  # indices[-valid_size:]

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

    print(f"White horse precision metric: {TP_validation / (TP_validation + FP_validation):.6f}")
