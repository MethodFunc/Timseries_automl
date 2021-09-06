from setting import define_parser
from utils.dataload import LoadDataframe
from utils.metrics import DataEval, scaling
from utils.preprocessing import preprocessing_data
from utils.torch_model.bid_lstm import BidLstm
from utils.torch_train import trainer, validation, tester
from utils.torch_setting import loss_fn, optim_fn
from wandb_setting import sweap_setting
from southwest_module import cleanup_df
from utils.plot import predict_plot

# from utils.all import *

import wandb
import time
import os
import torch
import numpy as np

"""
auth: methodfunc - Kwak Piljong
date: 2021.09.06
describe: use automl with wandb
"""

device = "cuda" if torch.cuda.is_available() else "cpu"


def default_train():

    config_defaults = {
        "epochs": 30,
        "batch_size": 128,
        "learning_rate": 1e-2,
        "optimizer": "adam",
        "num_layers": 1,
        "dropout": 0.2,
        "hidden_size": 64,
    }

    # wandb login & initializer

    wandb.init(config=config_defaults)
    wandb.login(key=args.api_key)

    config = wandb.config

    # load & data preprocessing
    trainloader, valloader, testloader = preprocessing_data(data, config, args)

    # Load model & parameter setting
    model = BidLstm(
        input_dim=args.features_size, output_size=args.output, config=config
    )

    model = model.to(device)

    loss = loss_fn(args.loss_fn)
    optim = optim_fn(model, config.optimizer, learning_rate=config.learning_rate)

    # Record losses
    train_losses = []
    val_losses = []

    # Train and validation
    times = 0.0
    for epoch in range(config.epochs):
        ts = time.time()
        model, train_loss = trainer(model, trainloader, optim, loss)
        val_loss = validation(model, valloader, loss)
        te = time.time()
        times += te - ts

        train_loss = train_loss / len(trainloader)
        val_loss = val_loss / len(valloader)
        print(
            f"[{epoch + 1}/{config.epochs}] - {te - ts:.2f}sec, train_loss:{train_loss:.4f}, val_loss: {val_loss:.4f}"
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        wandb.log({"loss": train_loss, "val_loss": val_loss})

    # log test_data wandb
    y_true, y_pred = tester(model, testloader)

    # calc evaluation data
    mse, rmse, mae, mape, acc = DataEval(y_true, y_pred).get()

    # Table create
    columns = ["MSE", "RMSE", "MAE", "MAPE", "ACC_MAX", "ACC_MEAN"]
    predict_table = wandb.Table(columns=columns)
    predict_table.add_data(mse, rmse, mae, mape, np.max(acc), np.mean(acc))

    # plot save local device root
    if not os.path.isdir(args.output_plot):
        os.mkdir(args.output_plot)

    # inverse target_data scale -> real data
    y_true_invert = tscale.inverse_transform(y_true.reshape(-1, 1))
    y_pred_invert = tscale.inverse_transform(y_pred.reshape(-1, 1))

    # plot setting
    ax = predict_plot(
        wandb,
        y_true_invert,
        y_pred_invert,
        view_count=args.view_count,
        method=args.method,
    )
    ax.savefig(f"./plot/{str(wandb.run.name)}.png")

    # log wandb with training data & predict_data
    wandb.log(
        {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "acc_min": np.min(acc),
            "acc_max": np.max(acc),
            "acc_mean": np.mean(acc),
            "predict_table": predict_table,
            "predict_plot": ax,
        }
    )


if __name__ == "__main__":
    args = define_parser()
    # Load Data & Clean data
    raw_data = LoadDataframe(args.path).get_df()
    features, targets = cleanup_df(raw_data)

    # Scaling
    fscale, f_data = scaling(features)
    tscale, t_data = scaling(targets)
    data = f_data, t_data
    args.features_size = f_data.shape[-1]

    # wandb sweap init(grid search, random search)
    sweap = sweap_setting()

    # project argument is project name. if you insert new name “wandb” can be creating new project with new name
    sweep_id = wandb.sweep(sweap, project=args.project_name)
    wandb.agent(sweep_id, default_train)
