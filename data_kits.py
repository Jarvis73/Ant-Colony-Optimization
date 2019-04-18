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
import collections
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def try_to_find(name):
    if (Path(__file__).parent / "data" / name).exists():
        obj = Path(__file__).parent / "data" / name
    elif (Path(__file__).parent / name).exists():
        obj = str(Path(__file__).parent / name)
    elif Path(name).exists():
        obj = name
    else:
        raise FileNotFoundError("Can not find {}".format(name))
    return obj


def load_txt(name):
    """
    TXT file only contains the number of cities and distance matrix.

    For example:
    # TSP4.txt
    4
    0 30 6 4
    30 0 5 10
    6 5 0 20
    4 10 20 0
    """
    with open(try_to_find(name)) as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    distances = np.zeros(shape=(n, n), dtype=np.int32)
    for i in range(n):
        distances[i] = [int(x) for x in lines[i + 1].strip().split()]
    print(n)
    return [str(x) for x in range(n)], distances


class TSPSpec(
    collections.namedtuple(
        "TSPBase", ["name", "comment", "dtype", "dimension", "edge_weight_type",
                    "data", "solutions"])):
    def __new__(cls, name, comment, dtype, dimension, edge_weight_type, data, solutions):
        return super(TSPSpec, cls).__new__(
            cls,
            name=name,
            comment=comment,
            dtype=dtype,
            dimension=dimension,
            edge_weight_type=edge_weight_type,
            data=np.array(data),
            solutions=np.array(solutions, dtype=np.int32)
        )

    def distance_matrix(self):
        if self.data.shape[0] == 0:
            return np.array([], dtype=np.float32)
        data = np.array(self.data)
        distance = np.sqrt(((data[:, None, :] - data[None, :, :]) ** 2).sum(axis=-1))
        return distance


def load_tsp(tsp_name, sln_names=None):
    """
    TSP file is the standard TSPLIB format
    """
    with open(try_to_find(tsp_name)) as f:
        lines = f.readlines()

    def strip(x):
        return x.split(":")[1].strip()

    name = ""
    comment = ""
    dtype = ""
    dimension = 0
    edge_weight_type = ""
    data = []
    for line in lines:
        if line.startswith("EOF"):
            break
        if line[0].isdecimal():
            data.append([float(x) for x in line.split()[1:]])
        if line.startswith("NAME"):
            name = strip(line)
        if line.startswith("COMMENT"):
            comment += strip(line) + "\n"
        if line.startswith("TYPE"):
            dtype = strip(line)
        if line.startswith("DIMENSION"):
            dimension = int(strip(line))
        if line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = strip(line)
    print("Read TSP file finished!")

    solutions = []
    if sln_names:
        if isinstance(sln_names, str):
            sln_names = [sln_names]
        for sln_name in sln_names:
            with open(try_to_find(sln_name)) as f:
                lines = f.readlines()

            path = []
            for line in lines:
                if line.startswith("EOF"):
                    break
                if line[0].isdecimal():
                    path.append(int(line.strip()))
                if line.startswith("-"):
                    path.append(path[0])
                    break
            if path:
                path = np.asarray(path, dtype=np.int32)
                solutions.append(path - path.min())
                print("Read a TSP solution finished!")

    tsp_spec = TSPSpec(name, comment, dtype, dimension, edge_weight_type, data, solutions)

    assert len(tsp_spec.data) == tsp_spec.dimension, \
        "Wrong length: data({}) vs dimension({})".format(len(tsp_spec.data), tsp_spec.dimension)

    return tsp_spec


def load(tsp_name, sln_names=None):
    if tsp_name.endswith("txt"):
        raise NotImplementedError
        # return load_txt(tsp_name)
    elif tsp_name.endswith("tsp"):
        return load_tsp(tsp_name, sln_names)
    else:
        ValueError("Only support [txt, tsp] file format, got {}.".format(tsp_name))


def adj_path(path):
    jj = np.argmin(path)
    return path[jj:] + path[:jj + 1]


#########################################################################################
#
#   Plotting Tool Kits
#

def show_graph(tsp_spec):
    plt.scatter(tsp_spec.data[:, 0], tsp_spec.data[:, 1], c="blue", s=12)


def show_path(tsp_spec, paths):
    if paths is None or len(paths) == 0:
        raise ValueError("Missing paths")
    path = paths[0]
    plt.plot(tsp_spec.data[path, 0], tsp_spec.data[path, 1], "b")
    plt.scatter(tsp_spec.data[:, 0], tsp_spec.data[:, 1], c="r", s=50)
    plt.axis("off")


def show_iters(iter_costs):
    x = np.arange(len(iter_costs[0]))
    for iter_cost in iter_costs:
        plt.plot(x, iter_cost)
    plt.xlabel("Iteration")
    plt.ylabel("TSP cost")


class DynamicShow(object):
    def __init__(self, tsp_spec):
        self.tsp_sepc = tsp_spec
        self.min_cost = sys.maxsize
        self.len = 0
        self.xdata = []
        self.ydata = []
        plt.ion()

    def launch(self, args):
        self.fig, self.ax = plt.subplots(2, 2, figsize=(14, 10))
        gs = self.ax[1, 1].get_gridspec()
        for ax in self.ax[1, :]:
            ax.remove()
        self.axbig = self.fig.add_subplot(gs[1, :])

        self.ax[0, 0].scatter(self.tsp_sepc.data[:, 0], self.tsp_sepc.data[:, 1], c="r", s=50)
        self.tour, = self.ax[0, 0].plot([], [], "b")
        self.ax[0, 0].axis("off")

        self.cost, = self.axbig.semilogy(self.xdata, self.ydata, "b")
        self.min_line, = self.axbig.plot([], [], ":", color="0.5")
        self.axbig.set_xlabel("Iteration")
        self.axbig.set_ylabel("TSP Cost")
        self.t = self.axbig.text(0, 0, "")

        tmp = np.ones([self.tsp_sepc.dimension] * 2, dtype=np.float32)
        tmp[0, 0] = 0
        self.img = self.ax[0, 1].imshow(tmp, cmap="gray")

        self.fig.suptitle("{} Cities - {} - alpha={}, beta={}, rho={}"
                          .format(self.tsp_sepc.dimension, args.alg, args.alpha, args.beta, args.rho))
        self.fig.canvas.draw_idle()

    def running(self, path, cost, pheromone):
        if cost < self.min_cost:
            self.min_cost = cost
        self.xdata.append(self.len)
        self.ydata.append(cost)

        if self.len % 1 == 0:
            self.tour.set_xdata(self.tsp_sepc.data[path, 0])
            self.tour.set_ydata(self.tsp_sepc.data[path, 1])
            self.cost.set_xdata(self.xdata)
            self.cost.set_ydata(self.ydata)
            self.min_line.set_xdata([0, self.len])
            self.min_line.set_ydata([self.min_cost, self.min_cost])
            min_ = pheromone.min()
            max_ = pheromone.max()
            if max_ != min_:
                self.img.set_data((pheromone - min_) / (pheromone.max() - min_))
            else:
                self.img.set_data(pheromone)

            self.t.set_position([0, self.min_cost])
            self.t.set_text("Min cost: {:.1f}".format(self.min_cost))

            self.axbig.relim()
            self.axbig.autoscale_view()

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

        self.len += 1
