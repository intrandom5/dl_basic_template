from dataset import cifar_dataset
from models import VGG16
from tqdm import tqdm
import argparse
import yaml
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main(config):
    train_dataset = cifar_dataset(train=True)
    test_dataset = cifar_dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = VGG16(input_shape=config["input_shape"], channels=config["channels"], dense_dims=config["dense_dims"], n_class=config["n_class"])
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config["lr"])

    writer = SummaryWriter(config["log_dir"])
    data = next(iter(train_loader))
    writer.add_graph(model, data["images"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_save_dir = config["model_save_dir"]
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model.to(device)
    for epoch in range(config["epochs"]):
        print("epoch :", epoch)
        model.train()
        train_loss = 0
        for data in tqdm(train_loader):
            data = {k: v.to(device) for k, v in data.items()}
            pred = model(data["images"])
            loss = criterion(pred, data["labels"])
            train_loss += loss.detach().cpu().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)
        print("train loss :", train_loss)
        writer.add_scalar('training loss', train_loss, epoch)

        model.eval()
        eval_loss = 0
        for data in tqdm(test_loader):
            data = {k: v.to(device) for k, v in data.items()}
            with torch.no_grad():
                pred = model(data["images"])
            loss = criterion(pred, data["labels"])
            eval_loss += loss.detach().cpu().item()

        print("eval loss :", eval_loss/len(test_loader))
        writer.add_scalar('eval loss', eval_loss, epoch)
        torch.save(model.state_dict, os.path.join(model_save_dir, f"epoch{epoch}.pt"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=str, help="실험 config 경로")
    config = parser.parse_args()
    with open(config.conf, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config["input_shape"] = eval(config["input_shape"])
    main(config)
