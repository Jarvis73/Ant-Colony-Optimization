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
    def __init__(self, edges_mat, dtype=np.int32, args=None):
        edges_mat = np.asarray(edges_mat, dtype)
        if edges_mat.shape[0] != edges_mat.shape[1]:
            raise ValueError("edges_mat must be a square matrix")

        self.args = args
        super(TSP, self).init(edges_mat.shape[0], edges_mat)

        bst_path = [x - 1 for x in [1, 14, 6, 18, 9, 7, 13, 15, 11, 17, 3, 5,
                                    12, 4, 2, 8, 19, 16, 10, 20, 1]]
        # bst_path = [x - 1 for x in [1, 7, 9, 4, 10, 6, 5, 3, 2, 8, 1]]
        bst_cost = self._edges_mat[bst_path[:-1], bst_path[1:]].sum()
        print("Best cost: {}\nBest path: {}\n".format(bst_cost, bst_path[:-1]))

        self.lock = threading.Lock()
        self.avg_val = None

    def avg(self):
        if self.avg_val is None:
            self.avg_val = np.mean(self._edges_mat)
        return self.avg_val

    def nn_cost(self, verbose=False):
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
        return min_cost
