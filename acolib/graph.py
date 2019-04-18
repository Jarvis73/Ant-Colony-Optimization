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
import threading
import numpy as np

import data_kits


class GraphBase(object):
    def init(self, num_nodes, edges_mat):
        self._num_nodes = num_nodes
        self._edges_mat = edges_mat

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def edges(self):
        return self._edges_mat


class TSP(GraphBase):
    def __init__(self, tsp_spec, args=None):
        if not isinstance(tsp_spec, data_kits.TSPSpec):
            raise TypeError("tsp_spec must be a TSPSpec instance")

        edges_mat = tsp_spec.distance_matrix()

        self.args = args
        super(TSP, self).init(edges_mat.shape[0], edges_mat)

        if len(tsp_spec.solutions) > 0:
            bst_path = tsp_spec.solutions[0]
            bst_cost = self._edges_mat[bst_path[:-1], bst_path[1:]].sum()
            print("Objective solution cost: {}\n".format(bst_cost))

        self.lock = threading.Lock()
        self.avg_val = None

        self.nn_path, self.nn_cost = self._nn_cost()

    def avg(self):
        if self.avg_val is None:
            self.avg_val = np.mean(self._edges_mat)
        return self.avg_val

    def _nn_cost(self, verbose=False):
        min_cost = sys.maxsize
        min_path = None
        for start in range(self.num_nodes):
            unseen = list(range(self.num_nodes))
            del unseen[start]
            cur = start
            cur_path = [start]
            cost = 0
            for _ in range(self.num_nodes - 1):
                min_cost_idx = int(np.argmin(self._edges_mat[cur, unseen]))
                next_node = unseen[min_cost_idx]
                del unseen[min_cost_idx]
                cur_path.append(next_node)
                cost += self._edges_mat[cur, next_node]
                cur = next_node
            cost += self._edges_mat[cur_path[-1], cur_path[0]]
            if cost < min_cost:
                min_cost = cost
                min_path = cur_path
        if verbose:
            print("NN cost:", min_cost, "NN path:", min_path)
        return min_path, min_cost
