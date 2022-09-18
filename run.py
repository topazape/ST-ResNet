import argparse
import random
import sys
from pathlib import Path

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(Path.joinpath(current_dir, "../")))

import torch.nn as nn
import torch.optim as optim
from stresnet import Trainer
from stresnet.dataset.makedataset import get_dataset
from stresnet.models import STResNet
from stresnet.utils import Cfg
from torch.utils.data import DataLoader


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "-s", "--seed", default=0, type=int, help="seed for initializing training"
    )

    return parser.parse_args()


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = make_parser()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = args.config_file
    config_file_dir = str(Path(config_file).resolve().parent)

    config = Cfg(config_file)
    dataset_params = config.get_params(type="dataset")
    model_params = config.get_params(type="model")
    learning_params = config.get_params(type="learning")

    train_dataset, valid_dataset, scaler, external_dim = get_dataset(
        data_files=dataset_params["data_files"],
        holiday_file=dataset_params["holiday_file"],
        meteorol_file=dataset_params["meteorol_file"],
        T=dataset_params["T"],
        len_closeness=dataset_params["len_closeness"],
        len_period=dataset_params["len_period"],
        len_trend=dataset_params["len_trend"],
        period_interval=dataset_params["period_interval"],
        trend_interval=dataset_params["trend_interval"],
        len_test=dataset_params["len_test"],
        use_meta=dataset_params["use_meta"],
        map_height=model_params["map_height"],
        map_width=model_params["map_width"],
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=learning_params["batch_size"],
        shuffle=False,
        drop_last=False,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=dataset_params["len_test"],
        shuffle=False,
        drop_last=False,
    )

    st_resnet = STResNet(
        len_closeness=dataset_params["len_closeness"],
        len_period=dataset_params["len_period"],
        len_trend=dataset_params["len_trend"],
        external_dim=external_dim,
        nb_flow=model_params["nb_flow"],
        map_height=model_params["map_height"],
        map_width=model_params["map_width"],
        nb_residual_unit=model_params["nb_residual_unit"],
    )
    st_resnet.to(args.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(st_resnet.parameters(), lr=learning_params["learning_rate"])
    trainer = Trainer(
        epochs=learning_params["epochs"],
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=args.device,
        save_dir=config_file_dir,
    )

    print(
        "SEED: {}, DEVICE: {}, nb_resunit: {}, closeness: {},  period: {}, trend: {}".format(
            args.seed,
            args.device,
            model_params["nb_residual_unit"],
            dataset_params["len_closeness"],
            dataset_params["len_period"],
            dataset_params["len_trend"],
        )
    )
    trainer.fit(st_resnet)


if __name__ == "__main__":
    main()
