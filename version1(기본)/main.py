from dataset import cifar_dataset
from models import VGG16
from tqdm import tqdm
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def main():
    train_dataset = cifar_dataset(train=True)
    test_dataset = cifar_dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = VGG16(input_shape=(32, 32), channels=[64, 128, 256, 512], dense_dims=[1000], n_class=10)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-5)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_save_dir = "./saved_models"
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model.to(device)
    for epoch in range(5):
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
        print("train loss :", train_loss/len(train_loader))

        model.eval()
        eval_loss = 0
        for data in tqdm(test_loader):
            data = {k: v.to(device) for k, v in data.items()}
            with torch.no_grad():
                pred = model(data["images"])
            loss = criterion(pred, data["labels"])
            eval_loss += loss.detach().cpu().item()

        print("eval loss :", eval_loss/len(test_loader))
        torch.save(model.state_dict, os.path.join(model_save_dir, f"epoch{epoch}.pt"))
    

if __name__ == "__main__":
    main()
