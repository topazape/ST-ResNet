import json
from configparser import ConfigParser
from dataclasses import dataclass


@dataclass
class Cfg:
    config_file: str

    def __post_init__(self):
        self.config = ConfigParser()
        with open(self.config_file, "r") as f:
            self.config.read_file(f)

    def get_params(self, type: str):
        if type == "dataset":
            return self._dataset_params()
        elif type == "model":
            return self._model_params()
        else:
            return self._learning_params()

    def _dataset_params(self):
        return {
            "data_files": json.loads(self.config.get("dataset", "data_files")),
            "holiday_file": self.config.get("dataset", "holiday_file"),
            "meteorol_file": self.config.get("dataset", "meteorol_file"),
            "T": self.config.getint("dataset", "T"),
            "len_closeness": self.config.getint("dataset", "len_closeness"),
            "len_period": self.config.getint("dataset", "len_period"),
            "len_trend": self.config.getint("dataset", "len_trend"),
            "period_interval": self.config.getint("dataset", "period_interval"),
            "trend_interval": self.config.getint("dataset", "trend_interval"),
            "len_test": self.config.getint("dataset", "len_test"),
            "use_meta": self.config.getboolean("dataset", "use_meta"),
        }

    def _model_params(self):
        return {
            "nb_flow": self.config.getint("model", "nb_flow"),
            "map_height": self.config.getint("model", "map_height"),
            "map_width": self.config.getint("model", "map_width"),
            "nb_residual_unit": self.config.getint("model", "nb_residual_unit"),
        }

    def _learning_params(self):
        return {
            "epochs": self.config.getint("learning", "epochs"),
            "batch_size": self.config.getint("learning", "batch_size"),
            "learning_rate": self.config.getfloat("learning", "learning_rate"),
        }
