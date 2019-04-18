# Copyright 2019 Jianwei Zhang All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =================================================================================

import sys
import tqdm
import argparse
import numpy as np
import matplotlib.pyplot as plt

import config
import data_kits
from acolib import graph
from acolib import colony


def main():
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    args = parser.parse_args()
    config.fill_default(args)
    tsp_spec = data_kits.load(args.tsp_file, args.sln_file)
    # data_kits.show_path(tsp_spec, tsp_spec.solutions)
    # return
    args.ants = config.maybe_fill(args.ants, tsp_spec.dimension)
    print(args)

    num_ants = args.ants
    num_iters = args.iters
    num_rep = args.repeat

    g = graph.TSP(tsp_spec, args=args)

    best_path = None
    best_cost = sys.maxsize
    best_cost_iters = []

    # Animation
    player = data_kits.DynamicShow(tsp_spec)
    player.launch(args)

    # Main loop
    for _ in tqdm.tqdm(range(num_rep), ascii=True):
        ant_col = colony.AntColony(g, num_ants, num_iters, args)
        ant_col.begin(player)
        if ant_col.best_path_cost < best_cost:
            best_path = ant_col.best_path
            best_cost = ant_col.best_path_cost
            best_cost_iters.append(ant_col.iter_costs)

    best_path = data_kits.adj_path(best_path)
    print("Best path:", best_path)
    print("Best cost:", best_cost)

    # plt.subplot(121)
    # data_kits.show_path(tsp_spec, [best_path])
    # plt.subplot(122)
    # data_kits.show_iters(best_cost_iters)
    #
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
