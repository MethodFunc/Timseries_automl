from torch import nn, optim


def loss_fn(loss_name):
    if loss_name == "mse":
        loss = nn.MSELoss()

    elif loss_name == "huber":
        loss = nn.HuberLoss()
    else:
        raise f"{loss_name} func not support"
    return loss


def optim_fn(model, optim_name, learning_rate):
    if optim_name == "adam":
        optims = optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_name == "sgd":
        optims = optim.SGD(model.parameters(), lr=learning_rate)

    elif optim_name == "rmsprop":
        optims = optim.RMSprop(model.parameters(), lr=learning_rate)

    else:
        raise f"{optim_name} func not support"

    return optims
