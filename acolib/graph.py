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
    def __init__(self, edges_mat, dtype=np.int32):
        edges_mat = np.asarray(edges_mat, dtype)
        if edges_mat.shape[0] != edges_mat.shape[1]:
            raise ValueError("edges_mat must be a square matrix")

        super(TSP, self).init(edges_mat.shape[0], edges_mat)

        self.lock = threading.Lock()
        self.avg_val = None

    def avg(self):
        if self.avg_val is None:
            self.avg_val = np.mean(self.edges_mat)
        return self.avg_val
