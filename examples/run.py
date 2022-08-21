import sys
from pathlib import Path
from typing import Any, Optional

import torch

current_dir = Path(__file__).resolve().parent
sys.path.append(str(Path.joinpath(current_dir, "../")))

import torch.nn as nn
import torch.optim as optim
from stresnet import Trainer
from stresnet.dataset import TaxiBJ
from stresnet.models import STResNet
from stresnet.utils import Cfg
from torch.utils.data import DataLoader, TensorDataset


def get_dataset(
    len_closeness: int,
    len_period: int,
    len_trend: int,
    period_interval: int,
    trend_interval: int,
    len_test: int,
) -> tuple[TensorDataset, TensorDataset, Any, Optional[int]]:
    taxibj = TaxiBJ()
    datasets = taxibj.create_dataset(
        len_closeness=len_closeness,
        len_period=len_period,
        len_trend=len_trend,
        period_interval=period_interval,
        trend_interval=trend_interval,
        len_test=len_test,
    )
    tr_X = datasets["X_train"]
    tr_y = datasets["y_train"]
    va_X = datasets["X_test"]
    va_y = datasets["y_test"]
    scaler = datasets["scaler"]
    external_dim = datasets.get("external_dim")

    train_dataset = TensorDataset(
        *(
            [torch.tensor(tr, dtype=torch.float32) for tr in tr_X]
            + [torch.tensor(tr_y, dtype=torch.float32)]
        )
    )
    valid_dataset = TensorDataset(
        *(
            [torch.tensor(va, dtype=torch.float32) for va in va_X]
            + [torch.tensor(va_y, dtype=torch.float32)]
        )
    )
    return (train_dataset, valid_dataset, scaler, external_dim)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_file = "./examples/L4-C3-P1-T1/config.ini"
    config_file_dir = str(Path(config_file).resolve().parent)

    config = Cfg(config_file)
    dataset_params = config.get_params(type="dataset")
    model_params = config.get_params(type="model")
    learning_params = config.get_params(type="learning")

    train_dataset, valid_dataset, scaler, external_dim = get_dataset(
        len_closeness=dataset_params["len_closeness"],
        len_period=dataset_params["len_period"],
        len_trend=dataset_params["len_trend"],
        period_interval=dataset_params["period_interval"],
        trend_interval=dataset_params["trend_interval"],
        len_test=dataset_params["len_test"],
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
    st_resnet.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(st_resnet.parameters(), lr=learning_params["learning_rate"])
    trainer = Trainer(
        epochs=learning_params["epochs"],
        train_loader=train_dataloader,
        valid_loader=valid_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        save_dir=config_file_dir,
    )

    print(
        "DEVICE: {}, nb_resunit: {}, closeness: {},  period: {}, trend: {}".format(
            device,
            model_params["nb_residual_unit"],
            dataset_params["len_closeness"],
            dataset_params["len_period"],
            dataset_params["len_trend"],
        )
    )
    trainer.fit(st_resnet)
