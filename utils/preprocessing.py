__all__ = ["preprocessing_data"]

from torch.utils.data import DataLoader
from .makeset import split_data, DataMaker

# 데이터를 나누고, torch의 Tensor형식으로 변환합니다.
def preprocessing_data(data, config, args):
    if not args.single:
        f_data, t_data = data
        x_train, x_val, x_test = split_data(
            f_data,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
        )
        y_train, y_val, y_test = split_data(
            t_data,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
        )

    else:
        single_data = data
        x_train, x_val, x_test = split_data(
            single_data,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
        )
        y_train, y_val, y_test = split_data(
            single_data,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
        )

    train_set = DataMaker(
        x=x_train, y=y_train, window_size=config.batch_size, sliding=args.sliding_func
    )
    val_set = DataMaker(
        x=x_val, y=y_val, window_size=config.batch_size, sliding=args.sliding_func
    )
    test_set = DataMaker(
        x=x_test, y=y_test, window_size=config.batch_size, sliding=args.sliding_func
    )

    train = DataLoader(
        train_set,
        batch_size=config.batch_size,
        drop_last=args.drop_last,
        shuffle=args.train_shuffle,
    )
    val = DataLoader(
        val_set,
        batch_size=config.batch_size,
        drop_last=args.drop_last,
        shuffle=args.val_shuffle,
    )
    test = DataLoader(
        test_set,
        batch_size=config.batch_size,
        drop_last=args.drop_last,
        shuffle=args.test_shuffle,
    )

    return train, val, test
