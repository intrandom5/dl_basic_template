from dataset import cifar_dataset
from models import VGG16
from tqdm import tqdm
import argparse
import os

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader


def main(args):
    train_dataset = cifar_dataset(train=True)
    test_dataset = cifar_dataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = VGG16(input_shape=args.input_shape, channels=args.channels, dense_dims=args.dense_dims, n_class=args.n_class)
    print(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_save_dir = args.model_save_dir
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)

    model.to(device)
    for epoch in range(args.epochs):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=str, help="VGG16의 conv layer들의 채널 수 리스트")
    parser.add_argument("--dense_dims", type=str, help="VGG16의 dense layer들의 채널 수 리스트")
    parser.add_argument("--n_class", type=int, help="데이터셋의 라벨 클래스 수")
    parser.add_argument("--input_shape", type=str, help="모델에 입력되는 이미지 크기")
    parser.add_argument("--model_save_dir", type=str, help="모델 저장 경로")
    parser.add_argument("--batch_size", type=int, help="배치 크기")
    parser.add_argument("--lr", type=float, help="learning rate 설정")
    parser.add_argument("--epochs", type=int, help="학습 에포크 설정")
    args = parser.parse_args()

    args.channels = eval(args.channels)
    args.dense_dims = eval(args.dense_dims)
    args.input_shape = eval(args.input_shape)

    main(args)
