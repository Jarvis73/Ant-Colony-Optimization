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
import time
import tqdm
import random
import threading
import numpy as np

from . import ant


class AntColony(object):
    def __init__(self, graph, num_ants, max_iters, args):
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
        self.args = args

        self._iter = -1
        self._ants = []

        self.alpha = self.args.alpha
        self.beta = self.args.beta
        self.rho = self.args.rho
        self.best_path_cost = sys.maxsize
        self.best_path = None
        self.last_best_iter = self._iter
        self.iter_best_path_cost = sys.maxsize
        self.iter_best_path = None
        self.iter_all_paths = []
        self.iter_all_costs = []

        if self.args.alg == "AS":
            self.pheromone_init = 1.0 * num_ants / self.graph.nn_cost
        elif self.args.alg == "EAS":
            self.pheromone_init = 2.0 * num_ants / (self.rho * self.graph.nn_cost)
        elif self.args.alg == "ASRank":
            self.pheromone_init = 0.5 * self.args.top * (self.args.top - 1) / (self.rho * self.graph.nn_cost)
        elif self.args.alg == "MMAS":
            self.pheromone_init = 1. / (self.rho * self.graph.nn_cost)
        elif self.args.alg == "ACS":
            self.pheromone_init = 1. / (self.graph.num_nodes * self.graph.nn_cost)
        else:
            raise ValueError("Wrong algorithm: {}".format(self.args.alg))
        self.pheromone = np.empty(self.graph.edges.shape, dtype=np.float32)
        self.pheromone.fill(self.pheromone_init)
        self.heuristic = np.zeros_like(self.graph.edges, dtype=np.float32)
        self.heuristic = np.divide(1., self.graph.edges, out=self.heuristic,
                                   where=self.graph.edges != 0)

        self.iter_costs = []

        # self.cond = threading.Condition()
        self.pheromone_lock = threading.Lock()
        self.solution_lock = threading.Lock()

        self.global_update = eval("self.global_update_" + self.args.alg)

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
        self.iter_best_path_cost = sys.maxsize
        self.iter_best_path = None
        self.iter_all_paths.clear()
        self.iter_all_costs.clear()
        # self.last_best_iter = self._iter

    def begin(self, player=None):
        self.player = player
        ids = [random.randint(0, self.graph.num_nodes - 1) for i in range(self._num_ants)]

        for ii in tqdm.tqdm(range(self._max_iters), ascii=True):
            self._iter = ii + 1
            self._ants = [ant.Ant(i, ids[i], self, local=self.args.alg == "ACS") for i in range(self._num_ants)]
            self.one_run(verbose=False)

            # There are some bugs when using threading.Condition(), so we use join() instead.
            for a in reversed(self._ants):
                a.join()
            # self.cond.acquire()
            # self.cond.wait()    # wait for all ants finish one iteration
            self.global_update()
            jj = np.argmin(self.best_path)
            self.player.running(self.best_path[jj:] + self.best_path[:jj + 1],
                                self.iter_best_path_cost,
                                self.pheromone)
            # self.cond.release()

    def one_run(self, verbose=False):
        self.avg_path_cost = 0
        self.ant_counter = 0
        self.reset()
        if verbose:
            print("Iteration", self._iter, " ", end="")
        for a in self._ants:
            # print("starting ant = %s" % a.id)
            a.start()

    def solution_update(self, a, verbose=False):
        """ Called by ants """
        self.solution_lock.acquire()
        # print("Update called by %s" % a.id)
        self.ant_counter += 1
        self.avg_path_cost += a.cost
        self.iter_all_paths.append(a.path)
        self.iter_all_costs.append(a.cost)

        if a.cost < self.iter_best_path_cost:
            self.iter_best_path_cost = a.cost
            self.iter_best_path = a.path
            # self.last_best_iter = self._iter

        if self.ant_counter == len(self._ants):
            self.avg_path_cost /= len(self._ants)
            if verbose:
                print("Current best (iter {}): {} cost: {} avg: {}".format(self.ant_counter, self.iter_best_path,
                                                                           self.iter_best_path_cost, self.avg_path_cost))
            # self.cond.acquire()
            # self.cond.notify()
            # self.cond.release()
        self.solution_lock.release()

    def maybe_update_best(self):
        self.iter_costs.append(self.iter_best_path_cost)
        if self.iter_best_path_cost < self.best_path_cost:
            self.best_path_cost = self.iter_best_path_cost
            self.best_path = self.iter_best_path

    def global_update_AS(self):
        self.maybe_update_best()
        after_evaporation = (1 - self.rho) * self.pheromone

        deposition = np.zeros_like(self.graph.edges, dtype=np.float32)
        for pa, co in zip(self.iter_all_paths, self.iter_all_costs):
            deposition[pa, pa[1:] + pa[:1]] += 1. / co

        self.pheromone_lock.acquire()
        self.pheromone = after_evaporation + deposition
        self.pheromone_lock.release()

    def global_update_EAS(self):
        self.maybe_update_best()
        after_evaporation = (1 - self.rho) * self.pheromone

        deposition = np.zeros_like(self.graph.edges, dtype=np.float32)
        for pa, co in zip(self.iter_all_paths, self.iter_all_costs):
            deposition[pa, pa[1:] + pa[:1]] += 1. / co
        deposition[self.best_path, self.best_path[1:] + self.best_path[:1]] += 1.0 / self.best_path_cost

        self.pheromone_lock.acquire()
        self.pheromone = after_evaporation + deposition
        self.pheromone_lock.release()

    def global_update_ASRank(self):
        self.maybe_update_best()
        after_evaporation = (1 - self.rho) * self.pheromone

        deposition = np.zeros_like(self.graph.edges, dtype=np.float32)
        rank = np.argsort(self.iter_all_costs)
        for ii in range(1, self.args.top):
            i = rank[ii - 1]
            deposition[self.iter_all_paths[i], self.iter_all_paths[i][1:] +
                       self.iter_all_paths[i][:1]] += 1. * (self.args.top - ii) / self.iter_all_costs[i]
        deposition[self.best_path, self.best_path[1:] +
                   self.best_path[:1]] += 1. * self.args.top / self.best_path_cost

        self.pheromone_lock.acquire()
        self.pheromone = after_evaporation + deposition
        self.pheromone_lock.release()

    def global_update_MMAS(self):
        self.maybe_update_best()
        after_evaporation = (1 - self.rho) * self.pheromone

        if self.args.local_search:
            if (26 <= self._iter <= 75 and self._iter % 5 == 0) or \
                    (76 <= self._iter <= 125 and self._iter % 3 == 0) or \
                    (126 <= self._iter <= 250 and self._iter % 2 == 0) or \
                    (self._iter > 250):
                chosen_best_cost = self.best_path_cost
                chosen_best_path = self.best_path
            else:
                chosen_best_cost = self.iter_best_path_cost
                chosen_best_path = self.iter_best_path
        elif self.args.iter_best_prob < 1.0 and random.random() > self.args.iter_best_prob:
            chosen_best_cost = self.best_path_cost
            chosen_best_path = self.best_path
        else:
            chosen_best_cost = self.iter_best_path_cost
            chosen_best_path = self.iter_best_path
        deposition = np.zeros_like(self.graph.edges, dtype=np.float32)
        deposition[chosen_best_path, chosen_best_path[1:] +
                   chosen_best_path[:1]] += 1. / chosen_best_cost

        if not hasattr(self, "tau_max"):
            self.tau_max = 1.0 / (self.rho * self.best_path_cost)
        if not hasattr(self, "tau_min"):
            tmp = 0.05 ** (1.0 / self.graph.num_nodes)
            self.tau_min = self.tau_max * (1 - tmp) / ((self.graph.num_nodes / 2. - 1) * tmp)

        self.pheromone_lock.acquire()
        self.pheromone = np.maximum(np.minimum(after_evaporation + deposition,
                                               self.tau_max), self.tau_min)
        self.pheromone_lock.release()

    def global_update_ACS(self):
        self.maybe_update_best()
        values = (1 - self.rho) * self.pheromone[self.best_path, self.best_path[1:] + self.best_path[:1]] + \
            self.rho / self.best_path_cost
        self.pheromone_lock.acquire()
        self.pheromone[self.best_path, self.best_path[1:] + self.best_path[:1]] = values
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
