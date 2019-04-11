# Ant Colony Optimization (Not finished!)

A python implementation of **Ant System** and its several variants for traditional Travelling Salesman Problem(TSP).

## Usage:

```bash
python main.py --tsp tsp_file --alg AS --ants 20
```

## Algorithms

| Name | Description |
|:----:|:-----------:|
| ACO  |蚁群算法基础框架: 信息素初始化, 蚂蚁寻路, 信息素局部/全局更新, 多线程并行|
| AS   |基础的蚁群算法, 包含信息素初始化, 信息素全局更新|
| EAS  |精英蚂蚁, 信息素全局更新时强化best-so-far路径|
| AS_rank |排序蚂蚁, 每次大迭代步中, 路径最短的前k只蚂蚁可以更新信息素|
| MMAS |最大-最小蚂蚁系统, 限制信息素的变化范围, 重点探索当前最优的路径, 信息素以小比率衰减|
| ACS | |

## Visualization

## TODO List

* Algorithms
  - [x] ACO Framework
  - [x] AS (Original: Ant System)
  - [x] EAS (Variant: Elitist Ant System)
  - [x] AS_rank (Variant: Rank-based Ant System)
  - [x] MMAS (Variant: Max-Min Ant System)
  - [ ] ACS (Extension: Ant Colony System)

* Visualization
  - [ ] TSP vis
  - [ ] Algorithm vis
  - [ ] Comparision
