from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path

import numpy as np
import tables
from sklearn.preprocessing import minmax_scale

current_dir = Path(__file__).resolve().parent


class Span(Enum):
    BJ13 = auto()
    BJ14 = auto()
    BJ15 = auto()
    BJ16 = auto()


@dataclass
class LoadData:
    BJ13: Path = Path("./TaxiBJ/BJ13_M32x32_T30_InOut.h5")
    BJ14: Path = Path("./TaxiBJ/BJ14_M32x32_T30_InOut.h5")
    BJ15: Path = Path("./TaxiBJ/BJ15_M32x32_T30_InOut.h5")
    BJ16: Path = Path("./TaxiBJ/BJ16_M32x32_T30_InOut.h5")
    METEOROLOGY: Path = Path("./TaxiBJ/BJ_Meteorology.h5")
    HOLIDAY: Path = Path("./TaxiBJ/BJ_Holiday.txt")

    def __post_init__(self):
        assert self.BJ13.exists()
        assert self.BJ14.exists()
        assert self.BJ15.exists()
        assert self.BJ16.exists()
        assert self.METEOROLOGY.exists()
        assert self.HOLIDAY.exists()

    def load_holiday(self, timeslots: np.ndarray) -> np.ndarray:
        timeslots = np.frompyfunc(lambda x: x[:8], 1, 1)(timeslots)

        with open(self.HOLIDAY, "r") as f:
            holidays = set([l.strip() for l in f])

        holidays = np.array(holidays)
        indices = np.where(np.isin(timeslots, holidays))[0]

        hv = np.zeros(len(timeslots))
        hv[indices] = 1.0
        return hv[:, np.newaxis]

    def load_meteorol(self, timeslots: np.ndarray):
        dat = tables.open_file(self.METEOROLOGY, mode="r")
        # dateformat: YYYYMMDD[slice]
        m_timeslots = dat.root.date.read().astype(str)
        wind_speed = dat.root.WindSpeed.read()
        weather = dat.root.Weather.read()
        temperature = dat.root.Temperature.read()
        dat.close()

        predicted_ids = np.where(np.isin(m_timeslots, timeslots))[0]
        cur_ids = predicted_ids - 1

        ws = wind_speed[cur_ids]
        wr = weather[cur_ids]
        te = temperature[cur_ids]

        # 0-1 scale
        ws = minmax_scale(ws)[:, np.newaxis]
        te = minmax_scale(te)[:, np.newaxis]

        # concatenate all these attributes
        merge_data = np.hstack([wr, ws, te])

        return merge_data

    def _remove_incomplete_days(
        self, dat: tables.file.File, T=48
    ) -> tuple[np.ndarray, np.ndarray]:
        # 20140425 has 24 timestamps, which does not appear in `incomplete_days` in the original implementation.
        # So I reimplemented it in a different way.
        data = dat.root.data.read()
        timestamps = dat.root.date.read().astype(str)

        dates, values = np.vstack(
            np.frompyfunc(lambda x: (x[:8], x[8:]), 1, 2)(timestamps)
        )
        # label encoding
        uniq_dates, labels = np.unique(dates, return_inverse=True)
        # groupby("labels")["values"].sum() != sum(range(1, 49))
        incomplete_days = uniq_dates[
            np.where(np.bincount(labels, values.astype(int)) != sum(range(1, (T + 1))))[
                0
            ]
        ]
        del_idx = np.where(np.isin(dates, incomplete_days))[0]
        new_data = np.delete(data, del_idx, axis=0)
        new_timestamps = np.delete(timestamps, del_idx)
        return new_data, new_timestamps

    def load_data(self, span: Enum):
        if span.name == "BJ13":
            dat = tables.open_file(self.BJ13, mode="r")
        elif span.name == "BJ14":
            dat = tables.open_file(self.BJ14, mode="r")
        elif span.name == "BJ15":
            dat = tables.open_file(self.BJ15, mode="r")
        else:
            dat = tables.open_file(self.BJ16, mode="r")

        data, timestamps = self._remove_incomplete_days(dat)
        dat.close()

        data[data < 0] = 0.0
        return data, timestamps
