from typing import Optional

import torch
import torch.nn as nn
from stresnet.models import ResUnit


class STResNet(nn.Module):
    def __init__(
        self,
        len_closeness: int,
        len_period: int,
        len_trend: int,
        external_dim: Optional[int],
        nb_flow: int,
        map_height: int,
        map_width: int,
        nb_residual_unit: int,
    ) -> None:
        super().__init__()
        self.external_dim = external_dim
        self.map_height = map_height
        self.map_width = map_width
        self.nb_flow = nb_flow
        self.nb_residual_unit = nb_residual_unit

        # models
        self.c_net = self._create_timenet(len_closeness)
        self.p_net = self._create_timenet(len_period)
        self.t_net = self._create_timenet(len_trend)
        if self.external_dim:
            # in/out flows * (len_closeness + len_period + len_trend)
            nb_total_flows = self.nb_flow * (len_closeness + len_period + len_trend)
            self.e_net = self._create_extnet(
                self.external_dim, nb_total_flows=nb_total_flows
            )

        # for fusion
        self.W_c = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_p = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )
        self.W_t = nn.parameter.Parameter(
            torch.randn(self.nb_flow, self.map_width, self.map_height),
            requires_grad=True,
        )

    def _create_extnet(self, ext_dim: int, nb_total_flows: int) -> nn.Sequential:
        ext_net = nn.Sequential(
            nn.Linear(ext_dim, nb_total_flows),
            nn.ReLU(inplace=True),
            # flatten in/out flow * grid_height * grid_width
            nn.Linear(nb_total_flows, self.nb_flow * self.map_height * self.map_width),
        )
        return ext_net

    def _create_timenet(self, length: int) -> nn.Sequential:
        time_net = nn.Sequential()
        time_net.add_module(
            "Conv1",
            nn.Conv2d(
                in_channels=(length * self.nb_flow),
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding="same",
            ),
        )

        for i in range(self.nb_residual_unit):
            time_net.add_module(
                f"ResUnit{i + 1}", ResUnit(in_channels=64, out_channels=64)
            )

        time_net.add_module(
            "Conv2",
            nn.Conv2d(
                in_channels=64, out_channels=2, kernel_size=3, stride=1, padding="same"
            ),
        )
        return time_net

    def forward(
        self,
        xc: torch.Tensor,
        xp: torch.Tensor,
        xt: torch.Tensor,
        ext: Optional[torch.Tensor],
    ) -> torch.Tensor:
        c_out = self.c_net(xc)
        p_out = self.p_net(xp)
        t_out = self.t_net(xt)

        if self.external_dim:
            e_out = self.e_net(ext).view(
                -1, self.nb_flow, self.map_width, self.map_height
            )
            # fusion with ext data
            res = self.W_c.unsqueeze(0) * c_out
            res += self.W_p.unsqueeze(0) * p_out
            res += self.W_t.unsqueeze(0) * t_out
            res += e_out
        else:
            res = self.W_c.unsqueeze(0) * c_out
            res += self.W_p.unsqueeze(0) * p_out
            res += self.W_t.unsqueeze(0) * t_out

        return torch.tanh(res)
