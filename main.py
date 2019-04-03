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

from acolib import graph
from acolib import colony


def main():
    # TODO(ZJW): Add argparser
    edges_mat = None
    num_ants = 20
    num_iters = 20
    num_rep = 1

    g = graph.TSP(edges_mat)

    best_path = None
    best_cost = sys.maxsize
    for _ in range(num_rep):
        ant_col = colony.AntColony(g, num_ants, num_iters)
        ant_col.begin()
        if ant_col.best_path_cost < best_cost:
            best_path = ant_col.best_path
            best_cost = ant_col.best_path_cost

    print("Best path: ", best_path)
    print("Best cost: ", best_cost)


if __name__ == "__main__":
    main()
