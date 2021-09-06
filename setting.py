import argparse


def define_parser():
    parser = argparse.ArgumentParser()
    args = parser.parse_args("")

    """
    Setting Parameters 
    """
    # Data가 들어있는 폴더 혹은 파일 지정
    args.path = "./data"
    args.output_plot = "./plot"

    # wandb setting
    args.project_name = "WT_Test"
    # data split size
    args.train_size = 0.8
    args.val_size = 0.1
    args.test_size = 0.1

    # data setting
    # if features data is same to target data single args True else False

    args.single = False
    args.window_size = 256
    args.sliding_func = True
    args.drop_last = True
    args.train_shuffle = False
    args.val_shuffle = False
    args.test_shuffle = False

    # support: minmax, normal, robust, standard
    args.method = "minmax"

    # Torch setting
    # support mse, huber
    args.loss_fn = "mse"

    # support adam, rmsprop, sgd
    args.optim = "adam"

    args.lr = 0.0001

    args.hidden_size = 512
    args.output = 1

    args.dropout = 0.3
    args.num_layers = 1

    args.epochs = 30

    args.view_count = -144
    args.method = "forward"

    return args
