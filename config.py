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


def add_arguments(parser):
    group = parser.add_argument_group(title="Global Arguments")
    group.add_argument("--tsp_file",
                       type=str,
                       required=True, help="TSP file. First line is the number of cities, and"
                                           "other lines store the distance matrix")
    group.add_argument("--alpha",
                       type=float,
                       default=1.0, help="Parameter to determine the pheromone influence")
    group.add_argument("--beta",
                       type=float,
                       default=2.0, help="Parameter to determine the heuristic influence")
    group.add_argument("--rho",
                       type=float, help="Evaporation rate")
    group.add_argument("--ants",
                       type=int, help="Number of ants")
    group.add_argument("--alg",
                       type=str,
                       choices=["AS", "EAS", "ASRank", "MMAS"],
                       required=True, help="Algorithm to perform")
    group.add_argument("--iters",
                       type=int,
                       default=20, help="Number of iterations")
    group.add_argument("--repeat",
                       type=int,
                       default=20, help="Number of repeat")

    group = parser.add_argument_group(title="ASRank Arguments")
    group.add_argument("--top",
                       type=int,
                       default=6, help="Used in ASRank. Top k ants to deposit pheromone")

    group = parser.add_argument_group(title="MMAS Arguments")
    group.add_argument("--iter_best_prob",
                       type=float,
                       default=1.0, help="Probability for using iter best path to update pheromone. "
                                         "The other choice is best-so-far path.")


def maybe_fill(x, v):
    if x is None:
        return v
    return x


def fill_default(args):
    if args.alg == "AS":
        args.rho = maybe_fill(args.rho, 0.5)
    elif args.alg == "EAS":
        args.rho = maybe_fill(args.rho, 0.5)
    elif args.alg == "ASRank":
        args.rho = maybe_fill(args.rho, 0.1)
    elif args.alg == "MMAS":
        args.rho = maybe_fill(args.rho, 0.2)
    else:
        raise ValueError
