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

import random
import threading
import numpy as np


class Ant(threading.Thread):
    def __init__(self, id_, start, colony, local=False):
        threading.Thread.__init__(self)
        self.id = id_
        self.start_node = start
        self.colony = colony

        self.cur_node = start
        self.graph = colony.graph
        self.path = [start]
        self.cost = 0
        self.exploit = 1.0
        self.rho = 0.99
        self.q0 = colony.args.q0
        self.xi = 0.1
        self.local = local  # local update

        self.num_nodes = self.graph.num_nodes
        self.unseen = list(range(self.graph.num_nodes))
        self.unseen.remove(self.start_node)

        if colony.args.alg in ["ACS"]:
            self.next = self.next_ACS
        else:
            self.next = self.next_AS

    def run(self):
        while not self.request_stop():
            self.graph.lock.acquire()
            next_node = self.next(self.cur_node)
            self.cost += self.graph.edges[self.cur_node, next_node]
            self.path.append(next_node)
            # print("Ant %s : %s, %s" % (self.id, self.path, self.cost,))
            if self.local:
                self.local_update(self.cur_node, next_node)
            self.graph.lock.release()

            self.cur_node = next_node

        self.cost += self.graph.edges[self.path[-1], self.path[0]]
        if self.cost < 700:
            self.local_search()
        self.colony.solution_update(self)

        # print("Ant thread %s terminating." % self.id)

    def next_AS(self, i):
        next_prob = (self.colony.pheromone[i, self.unseen] ** self.colony.alpha *
                     self.colony.heuristic[i, self.unseen] ** self.colony.beta)
        sum_ = next_prob.sum()
        # if sum_ == 0.:
        #     normed_next_prob = np.ones_like(next_prob, dtype=np.float32) / next_prob.shape[0]
        # else:
        normed_next_prob = next_prob / sum_
        next_node = random.choices(self.unseen, weights=normed_next_prob)[0]
        self.unseen.remove(next_node)
        return next_node

    def next_ACS(self, i):
        next_prob = (self.colony.pheromone[i, self.unseen] ** self.colony.alpha *
                     self.colony.heuristic[i, self.unseen] ** self.colony.beta)
        sum_ = next_prob.sum()
        if sum_ == 0.:
            normed_next_prob = np.ones_like(next_prob, dtype=np.float32) / next_prob.shape[0]
        else:
            normed_next_prob = next_prob / sum_
        if random.random() < self.q0:
            idx = np.argmax(next_prob)
            next_node = self.unseen[idx]
        else:
            next_node = random.choices(self.unseen, weights=normed_next_prob)[0]
        self.unseen.remove(next_node)
        return next_node

    def request_stop(self):
        return not self.unseen

    def local_update(self, i, j):
        val = (1 - self.xi) * self.colony.pheromone[i, j] + self.xi * self.colony.pheromone_init
        self.colony.atomic_update_pheromone(i, j, val)

    def local_search(self):
        # 2-exchange
        best_path = self.path
        best_cost = self.cost
        for i in range(self.num_nodes - 2):
            for j in range(i + 2, self.num_nodes):
                new_path = self.path[:i + 1] + list(reversed(self.path[i + 1:j + 1])) + self.path[j + 1:]
                new_cost = self.graph.edges[new_path, new_path[1:] + new_path[:1]].sum()
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost

        self.path = best_path
        self.cost = best_cost
