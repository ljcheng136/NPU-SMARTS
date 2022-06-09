# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.types import (
    Distribution,
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    TrafficActor,
)

normal = TrafficActor(
    name="car",
    speed=Distribution(mean=1, sigma=1.2),
    min_gap = Distribution(mean=3, sigma=1.2),
)

# flow_name = (start_lane, end_lane,)
route_opt = [
    (0, 0),
    (1, 1),
    (2, 2),
]

# Traffic combinations = 3C2 + 3C3 = 3 + 1 = 4
# Duplicate traffic = traffic_combinations * 50 = 200 
min_flows = 2
max_flows = 3
route_comb = [
    com * 50
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(route_opt, elems)
]

traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=("gneE3", r[0], 0),
                    end=("gneE4", r[1], "max"),
                ),
                # Random flow rate, between x and y vehicles per minute.
                rate=60 * random.uniform(6, 10),
                # Random flow start time, between x and y seconds.
                begin=random.uniform(0, 5),
                # For an episode with maximum_episode_steps=3000 and step
                # time=0.1s, maximum episode time=300s. Hence, traffic set to
                # end at 900s, which is greater than maximum episode time of
                # 300s.
                end=60 * 15,
                actors={normal: 1},
            )
            for r in routes
        ]
    )


ego_missions = [
    Mission(Route(begin=("gneE6", 0, 10), end=("gneE4", 2, "max")),start_time=15),
    Mission(Route(begin=("gneE3", 0, 10), end=("gneE4", 0, "max")),start_time=15),
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)