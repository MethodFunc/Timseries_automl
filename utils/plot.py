import matplotlib.pyplot as plt

plt.style.use("ggplot")


def train_plot(train, val):
    plt.style.use("ggplot")
    plt.figrue(figsize=(20, 6))
    plt.plot(train, label="train loss")
    plt.plot(val, label="val loss")
    plt.legend()
    plt.show()


def predict_plot(wandb, y_true, y_val, view_count, method="forward"):
    plt.style.use("ggplot")
    plt.figrue(figsize=(20, 6))

    if method == "forward":
        plt.plot(y_true[:view_count], label="actual")
        plt.plot(y_val[:view_count], label="predict")

    elif method == "backward":
        plt.plot(y_true[view_count:], label="actual")
        plt.plot(y_val[view_count:], label="predict")

    plt.ylabel("ActivePower")
    plt.title(f"{str(wandb.run.name)}_predict plot 144")
    plt.legend()

    return plt
