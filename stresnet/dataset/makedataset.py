from datetime import datetime
from typing import Any, Optional

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from stresnet.dataset.load_data import LoadData
from stresnet.dataset.stmatrix import STMatrix
from torch.utils.data import TensorDataset


def get_dataset(
    data_files: list[str],
    holiday_file: str,
    meteorol_file: str,
    T: int,
    len_closeness: int,
    len_period: int,
    len_trend: int,
    period_interval: int,
    trend_interval: int,
    len_test: int,
    use_meta: bool,
    map_height: int,
    map_width: int
) -> tuple[TensorDataset, TensorDataset, Any, Optional[int]]:
    mkdata = MakeDataset(
        data_files=data_files,
        holiday_file=holiday_file,
        meteorology_file=meteorol_file
    )

    datasets = mkdata.create_dataset(
        T=T,
        len_closeness=len_closeness,
        len_period=len_period,
        len_trend=len_trend,
        period_interval=period_interval,
        trend_interval=trend_interval,
        len_test=len_test,
        use_meta=use_meta,
        map_height=map_height,
        map_width=map_width
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


class MakeDataset(LoadData):
    def create_dataset(
        self,
        T: int,
        len_closeness: int,
        len_period: int,
        len_trend: int,
        period_interval: int,
        trend_interval: int,
        len_test: int,
        use_meta: bool,
        map_height: int,
        map_width: int
    ) -> dict[str, Any]:
        main_data = self._create_main_data(
            T,
            len_closeness,
            len_period,
            len_trend,
            period_interval,
            trend_interval,
            len_test,
            map_height,
            map_width
        )
        # swap axis from (n_samples, in/out-flow, grid-i, grid-j)
        #   to (n_samples, grid-i, grid-j in/out-flow)
        xc = main_data["closeness"]
        xp = main_data["period"]
        xt = main_data["trend"]
        y = main_data["y"]

        train_xc = xc[:-len_test]
        train_xp = xp[:-len_test]
        train_xt = xt[:-len_test]
        train_y = y[:-len_test]

        test_xc = xc[-len_test:]
        test_xp = xp[-len_test:]
        test_xt = xt[-len_test:]
        test_y = y[-len_test:]

        train_X = [train_xc, train_xp, train_xt]
        test_X = [test_xc, test_xp, test_xt]

        if self.HOLIDAY:
            use_holiday = True
        else:
            use_holiday = False
        
        if self.METEOROLOGY:
            use_meteorol = True
        else:
            use_meteorol = False

        if any([use_meta, use_holiday, use_meteorol]):
            meta_features = self._create_meta_data(
                use_meta, use_holiday, use_meteorol, timeslots=main_data["timestamps_y"]
            )
            train_meta = meta_features[:-len_test]
            train_X.append(train_meta)

            test_meta = meta_features[-len_test:]
            test_X.append(test_meta)

        timestamps_y = main_data["timestamps_y"]
        train_timestamps = timestamps_y[:-len_test]
        test_timestamps = timestamps_y[-len_test:]

        dataset = {
            "X_train": train_X,
            "X_test": test_X,
            "y_train": train_y,
            "y_test": test_y,
            "ts_train": train_timestamps,
            "ts_test": test_timestamps,
            "scaler": main_data["scaler"],
        }

        if len(dataset["X_train"]) == 4:
            dataset["external_dim"] = dataset["X_train"][-1].shape[1]

        return dataset

    def _create_meta_data(
        self,
        use_meta: bool,
        use_holiday: bool,
        use_meteorol: bool,
        timeslots: np.ndarray,
    ) -> np.ndarray:
        meta_features = []
        if use_meta:
            time_feature = self._ts2vec(timeslots=timeslots)
            meta_features.append(time_feature)
        if use_holiday:
            holiday_feature = self.load_holiday(timeslots=timeslots)
            meta_features.append(holiday_feature)
        if use_meteorol:
            meteorol_feature = self.load_meteorol(timeslots=timeslots)
            meta_features.append(meteorol_feature)

        meta_features = np.hstack(meta_features)
        return meta_features

    def _create_main_data(
        self,
        T: int,
        len_closeness: int,
        len_period: int,
        len_trend: int,
        period_interval: int,
        trend_interval: int,
        len_test: int,
        map_height: int,
        map_width: int
    ) -> dict[str, Any]:
        data_all, timestamp_all = self.load_data(T=T)
        data_train = np.vstack(data_all)[:-len_test]

        # scale [-1, 1]
        scaler = MinMaxScaler(feature_range=(-1, 1))
        # MinMaxScaler only accepts dim <= 2, so the data is reshaped and appled
        scaler.fit(data_train.reshape(-1, 1))
        # return to original shape
        data_all_scaled = [
            scaler.transform(data.reshape(-1, 1)).reshape(data.shape)
            for data in data_all
        ]

        # create st-matrices
        xc, xp, xt, y, ts_y = [], [], [], [], []
        for data, timestamps in zip(data_all_scaled, timestamp_all):
            stmtx = STMatrix(data, timestamps, T)
            d_mtx = stmtx.create_matrices_dict(
                len_closeness=len_closeness,
                len_period=len_period,
                len_trend=len_trend,
                period_interval=period_interval,
                trend_interval=trend_interval,
                map_height=map_height,
                map_width=map_width
            )
            xc.append(d_mtx["closeness"])
            xp.append(d_mtx["period"])
            xt.append(d_mtx["trend"])
            y.append(d_mtx["y"])
            ts_y.append(d_mtx["timestamps_y"])

        # len_closeness = 3, len_period = 1, len_trend = 1
        # Same shape as original implementation with same condition.
        # closeness: (15072, 6, 32, 32), period: (15072, 2, 32, 32)
        # trend: (15072, 2, 32, 32), y: (15072, 2, 32, 32), ts_y: (15072,)
        main_data = {
            "closeness": np.vstack(xc),
            "period": np.vstack(xp),
            "trend": np.vstack(xt),
            "y": np.vstack(y),
            "timestamps_y": np.hstack(ts_y),
            "scaler": scaler,
        }
        return main_data

    def _ts2vec(self, timeslots: np.ndarray) -> np.ndarray:
        ts_vec = np.frompyfunc(lambda x: datetime.strptime(x[:8], "%Y%m%d"), 1, 1)(
            timeslots
        )
        # weekday() range [0, 6], Monday is 0, Sunday is 6
        ts_wvec = np.frompyfunc(lambda x: x.weekday(), 1, 1)(ts_vec).astype(int)
        # to one-hot matrix
        # last item is a flag for weekday
        ret_vec = np.eye((np.unique(ts_wvec).size + 1))[ts_wvec]
        weekday_idx = np.where(ts_wvec < 5)[0]
        ret_vec[weekday_idx, -1] = 1

        return ret_vec
