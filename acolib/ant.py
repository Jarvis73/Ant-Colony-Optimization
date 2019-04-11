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


class Ant(threading.Thread):
    def __init__(self, id_, start, colony):
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

        self.unseen = list(range(self.graph.num_nodes))
        del self.unseen[start]

        self.next = self.next_AS

    def run(self):
        while not self.request_stop():
            self.graph.lock.acquire()
            next_node = self.next(self.cur_node)
            self.cost += self.graph.edges[self.cur_node, next_node]
            self.path.append(next_node)
            # print("Ant %s : %s, %s" % (self.id, self.path, self.cost,))
            # self.local_update(self.cur_node, next_node)
            self.graph.lock.release()

            self.cur_node = next_node

        self.cost += self.graph.edges[self.path[-1], self.path[0]]
        self.colony.solution_update(self)

        # print("Ant thread %s terminating." % self.id)

    def next_AS(self, i):
        next_prob = (self.colony.pheromone[i, self.unseen] ** self.colony.alpha *
                     self.colony.heuristic[i, self.unseen] ** self.colony.beta)
        idx = np.argmax(next_prob)
        next_node = self.unseen[idx]
        del self.unseen[idx]
        return next_node

    def request_stop(self):
        return not self.unseen

    def local_update(self, i, j):
        val = (1 - self.rho) * self.colony.pheromone[i, j] + self.rho * self.colony.pheromone_init
        self.colony.atomic_update_pheromone(i, j, val)
