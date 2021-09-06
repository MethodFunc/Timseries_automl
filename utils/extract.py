import numpy as np
import pandas as pd
import os
import datetime


def extract_result(y_true, y_pred, wandb, args, idx=None):
    """
    Save result
    """
    y_true = np.array(y_true) if not isinstance(y_true, np.ndarray) else y_true
    y_pred = np.array(y_pred) if not isinstance(y_pred, np.ndarray) else y_pred

    save_time = datetime.datetime.now()
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    plot_path = os.path.join(args.output_plot, f"{wandb.run.name}.png")

    if idx:
        df = pd.DataFrame(
            {"date": idx, "pred": y_pred, "true": y_true, "plot_path": plot_path}
        )
    else:
        df = pd.DataFrame({"pred": y_pred, "true": y_true, "plot_path": plot_path})

    if not os.path.isdir(args.output_csv):
        os.mkdir(args.output_csv)
    df.to_csv(
        f"{args.output_csv}/f{save_time.strftime('%Y%m%d_%H%M%S')}_result.csv",
        index=False,
    )
