import numpy as np
from tqdm import tqdm

# The original implementation uses a lot of loops, which slows down the execution time.
# Therefore, I reimplemented to reduce loops as much as possible.


class STMatrix:
    def __init__(self, data: np.ndarray, timestamps: np.ndarray, T: int = 48) -> None:
        assert len(data) == len(timestamps)
        self.data = data
        self.timestamps = timestamps
        self.T = T
        self.np_timestamps = self._string2timestamp(timestamps)
        self.ts_all = np.arange(
            self.np_timestamps.min(),
            (self.np_timestamps.max() + np.timedelta64(30, "m")),
            np.timedelta64(30, "m"),
        )

    def _string2timestamp(self, timestamps: np.ndarray) -> np.ndarray:
        time_per_slot = 24 / self.T
        num_per_T = self.T // 24
        ts = np.frompyfunc(
            lambda x: "{}-{}-{}T{:02d}:{:02d}".format(
                x[:4],
                x[4:6],
                x[6:8],
                int((int(x[8:]) - 1) * time_per_slot),
                ((int(x[8:]) - 1) % num_per_T) * int(60 * time_per_slot),
            ),
            1,
            1,
        )(timestamps)
        return ts.astype("datetime64")

    def get_indices(self, timestamps: np.ndarray) -> np.ndarray:
        indices = np.where(np.isin(self.np_timestamps, timestamps))[0]
        return indices

    def get_matrices(self, timestamps: np.ndarray) -> np.ndarray:
        indices = self.get_indices(timestamps)
        return self.data[indices, :, :, :]

    def create_matrices_dict(
        self,
        len_closeness: int = 3,
        len_period: int = 3,
        len_trend: int = 3,
        period_interval: int = 1,
        trend_interval: int = 7,
    ) -> dict[str, np.ndarray]:
        start_t = max(
            [
                len_closeness,
                period_interval * len_period * self.T,
                trend_interval * len_trend * self.T,
            ]
        )
        x_c, x_p, x_t, y, ts_y = [], [], [], [], []
        for t in tqdm(self.np_timestamps[start_t:], dynamic_ncols=True):
            idx = np.where(self.ts_all == t)[0]
            if idx.size != 0:
                idx = idx.item()
            else:
                continue

            # closeness
            c_ts = self.ts_all[(idx - len_closeness) : idx]
            # period
            p_ts = self.ts_all[
                (idx - self.T * period_interval * len_period) : idx : self.T
                * period_interval
            ]
            # trend
            t_ts = self.ts_all[
                (idx - self.T * trend_interval * len_trend) : idx : self.T
                * trend_interval
            ]

            if np.all(
                [
                    np.isin(c_ts, self.np_timestamps).all(),
                    np.isin(p_ts, self.np_timestamps).all(),
                    np.isin(t_ts, self.np_timestamps).all(),
                ]
            ):
                c_mtx = self.get_matrices(c_ts)
                c_mtx = c_mtx.reshape(-1, 32, 32)
                x_c.append(c_mtx)

                p_mtx = self.get_matrices(p_ts)
                p_mtx = p_mtx.reshape(-1, 32, 32)
                x_p.append(p_mtx)

                t_mtx = self.get_matrices(t_ts)
                t_mtx = t_mtx.reshape(-1, 32, 32)
                x_t.append(t_mtx)

                y.append(self.get_matrices(np.array([t])))
                ts_y.append(self.timestamps[self.get_indices(np.array([t]))])

        x_c = np.array(x_c)
        x_p = np.array(x_p)
        x_t = np.array(x_t)
        y = np.array(y).squeeze()
        ts_y = np.array(ts_y).squeeze()

        dataset = {
            "closeness": x_c,
            "period": x_p,
            "trend": x_t,
            "y": y,
            "timestamps_y": ts_y,
        }
        return dataset
