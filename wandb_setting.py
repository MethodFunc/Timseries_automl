def sweap_setting():
    sweep_config = {
        "method": "random",  # grid, random
        "metric": {"name": "loss", "goal": "minimize"},
        "parameters": {
            "epochs": {"values": [30, 50, 100]},
            "batch_size": {"values": [256, 144, 128, 64, 32, 72]},
            "dropout": {"values": [0.3, 0.4, 0.5]},
            "num_layers": {"values": [4, 3, 2, 1]},
            "learning_rate": {"values": [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]},
            "hidden_size": {"values": [128, 256, 512]},
            "optimizer": {"values": ["adam", "sgd", "rmsprop"]},
        },
    }

    return sweep_config
