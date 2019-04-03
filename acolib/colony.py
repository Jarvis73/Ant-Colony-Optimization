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
import random
import threading
import numpy as np

from . import ant


class AntColony(object):
    def __init__(self, graph, num_ants, max_iters):
        """
        Ant colony implementation

        Parameters
        ----------
        graph: TSP
            A graph instance
        num_ants: int
            Number of ants in ACO
        max_iters: int
            Maximum number of iterations
        """

        self.graph = graph
        self._num_ants = num_ants
        self._max_iters = max_iters

        self._iter = -1
        self._ants = []
        self.pheromone_init = 1. / (self.graph.num_nodes * 0.5 * self.graph.avg())
        self.pheromone = np.empty(self.graph.edges_mat.shape, dtype=np.float32)
        self.reset_pheromone()
        self.heuristic = 1. / self.graph.edges_mat

        self.alpha = 1.0
        self.beta = 1.0
        self.rho = 0.1
        self.best_path_cost = sys.maxsize
        self.best_path = None
        self.last_best_iter = self._iter

        self.cond = threading.Condition()
        self.pheromone_lock = threading.Lock()
        self.solution_lock = threading.Lock()

    @property
    def num_ants(self):
        return self._num_ants

    @property
    def max_iters(self):
        return self._max_iters

    @property
    def iter(self):
        return self._iter

    def reset(self):
        self.best_path_cost = sys.maxsize
        self.best_path = None
        self.last_best_iter = self._iter

    def begin(self):
        self._ants = [ant.Ant(i, random.randint(0, self.graph.num_nodes - 1), self)
                      for i in range(self._num_ants)]
        self._iter = 0

        while self._iter < self._max_iters:
            self.one_run()

            self.cond.acquire()
            self.cond.wait()    # wait for all ants finish one iteration

            self.pheromone_lock.acquire()
            self.global_update()
            self.pheromone_lock.release()
            self.cond.release()

    def one_run(self):
        self.avg_path_cost = 0
        self.ant_counter = 0
        self._iter += 1
        print("Iteration", self._iter)
        for a in self._ants:
            a.start()

    def solution_update(self, a):
        """ Called by ants """
        self.solution_lock.acquire()
        self.ant_counter += 1
        self.avg_path_cost += a.path_cost

        if a.path_cost < self.best_path_cost:
            self.best_path_cost = a.path_cost
            self.best_path = a.path
            self.last_best_iter = self._iter

        if self.ant_counter == len(self._ants):
            self.avg_path_cost /= len(self._ants)
            print("Current best (iter {}): {} cost: {} avg: {}".format(self._iter, self.best_path,
                                                                       self.best_path_cost, self.avg_path_cost))
            self.cond.acquire()
            self.cond.notify()
            self.cond.release()
        self.solution_lock.release()

    def global_update(self):
        best_path_mat = np.zeros_like(self.graph.edges_mat, dtype=np.float32)
        best_path_mat[self.best_path, self.best_path[1:] + self.best_path[:1]] = 1
        after_evaporation = (1 - self.rho) * self.pheromone
        deposition = self.rho / self.best_path_cost * best_path_mat
        self.pheromone_lock.acquire()
        self.pheromone = after_evaporation + deposition
        self.pheromone_lock.release()

    def atomic_update_pheromone(self, i, j, val):
        self.pheromone_lock.acquire()
        self.pheromone[i, j] = val
        self.pheromone_lock.release()

    def reset_pheromone(self):
        self.pheromone_lock.acquire()
        self.pheromone.fill(self.pheromone_init)
        self.pheromone_lock.release()

    def request_stop(self):
        return self._iter == self.max_iters
