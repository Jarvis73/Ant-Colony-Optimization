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
import pickle
import argparse
import numpy as np
from pathlib import Path

import config
from acolib import graph
from acolib import colony


def load(name):
    if name is None:
        name = "./data/citiesAndDistances.pickled"
    with open(name, "rb") as f:
        names, distances = pickle.load(f, encoding="latin1")
    distances = np.asarray(distances, dtype=np.int32)
    return names, distances


def try_to_find(name):
    obj = None
    if (Path(__file__).parent / "data" / name).exists():
        obj = Path(__file__).parent / "data" / name
    elif (Path(__file__).parent / name).exists():
        obj = str(Path(__file__).parent / name)
    elif Path(name).exists():
        obj = name
    else:
        raise FileNotFoundError("Can not find {}".format(name))
    return obj


def loadv2(name):
    with open(try_to_find(name)) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    distances = np.zeros(shape=(n, n), dtype=np.int32)
    for i in range(n):
        distances[i] = [int(x) for x in lines[i + 1].strip().split()]
    print(n)
    return [str(x) for x in range(n)], distances


def adj_path(path):
    jj = np.argmin(path)
    return path[jj:] + path[:jj]


def main():
    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    args = parser.parse_args()
    config.fill_default(args)
    names, edges_mat = loadv2(args.tsp_file)
    args.ants = config.maybe_fill(args.ants, len(names))
    print(args)

    num_ants = args.ants
    num_iters = args.iters
    num_rep = args.repeat

    g = graph.TSP(edges_mat, args=args)

    best_path = None
    best_cost = sys.maxsize
    for _ in range(num_rep):
        ant_col = colony.AntColony(g, num_ants, num_iters, args)
        ant_col.begin()
        if ant_col.best_path_cost < best_cost:
            best_path = ant_col.best_path
            best_cost = ant_col.best_path_cost

    best_path = adj_path(best_path)
    print("Best path:", best_path)
    print("Best path:", list(reversed(best_path)))
    print("Best cost:", best_cost)


if __name__ == "__main__":
    main()
