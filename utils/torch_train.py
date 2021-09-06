__all__ = ["trainer", "validation", "tester"]

import numpy as np
import torch

"""
Simple torch trainer and tester
auth: Methodfunc - Kwak Piljong
date: 2021.08.26
version: 0.1
"""

device = "cuda" if torch.cuda.is_available() else "cpu"

# 해당 변수가 Tensor타입인지 확인합니다.
def check_tensor(x):
    return torch.Tensor(x) if not isinstance(x, torch.Tensor) else x


def trainer(model, dataloader, optimizer, loss_fn):
    model.train()

    train_loss = 0.0
    for idx, (X, y) in enumerate(dataloader):
        X = check_tensor(X)
        y = check_tensor(y)

        X = X.to(device)
        y = y.to(device)

        model.zero_grad()
        optimizer.zero_grad()
        output = model(X)

        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return model, train_loss


def validation(model, dataloader, loss_fn):
    model.eval()

    val_loss = 0.0
    with torch.no_grad():
        for (X, y) in dataloader:
            X = check_tensor(X)
            y = check_tensor(y)

            X = X.to(device)
            y = y.to(device)

            output = model(X)
            loss = loss_fn(output, y)

            val_loss += loss.item()

    return val_loss


def tester(model, dataloader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for (X, y) in dataloader:
            X = check_tensor(X)
            y = check_tensor(y)

            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            y_true.append(y.cpu().detach().numpy())
            y_pred.append(pred.cpu().detach().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return y_true, y_pred
