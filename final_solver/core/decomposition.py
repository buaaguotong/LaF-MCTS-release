"""Decomposition adapter for the released LaF-MCTS final solver.

The logic below is extracted from the authors' original experimental source
and keeps the decomposition-to-subproblem conversion used during evaluation.
"""
from __future__ import annotations

import random
import numpy as np


class Decomposition:
    def __init__(self, decomposer_solver, data, sub_problem_size=15,
                 record_path_size=4, original_data=None, seed=None):
        if original_data is None:
            raise ValueError("original_data is required")
        self.decomposer_solver = decomposer_solver
        self.data = data
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.coordinates = original_data['coordinates']
        self.dist_matrix = original_data['dist_matrix']
        self.capacity = original_data['capacity']
        self.demands = original_data['demands']
        self.config = original_data['config']

        self.depot = self.data.depots()[0]
        self.depot_x = self.depot.x
        self.depot_y = self.depot.y
        self.centroids = []
        self.angles = None
        self.num_clients = self.data.num_clients
        self.record_path_size = record_path_size
        self.sub_problem_size = sub_problem_size

    def decompose(self, solution, vfreq):
        self.solution = solution
        self.vfreq = vfreq
        self.centroids = []
        routes = []
        for route in solution.routes():
            routes.append(route.visits())
            self.centroids.append(route.centroid())

        self.centroids = np.array(self.centroids)
        self.calculatebarycentre()

        subproblems, arcs = self.decompositer(
            routes=routes,
            centroids=self.centroids,
            angles=self.angles,
            dist_matrix=self.dist_matrix,
            vfreq=self.vfreq,
            num_clients=self.num_clients,
            sub_problem_size=self.sub_problem_size,
            record_path_size=self.record_path_size,
            seed=self.seed,
        )
        if subproblems is not None:
            self.check_subproblems(subproblems)
        return subproblems, arcs

    def calculatebarycentre(self):
        depot_x = self.depot.x
        depot_y = self.depot.y
        delta = self.centroids - np.array([depot_x, depot_y])
        self.angles = np.arctan2(delta[:, 1], delta[:, 0])

    def decompositer(self, routes, centroids, angles, dist_matrix, vfreq,
                     num_clients, sub_problem_size, record_path_size, seed):
        subproblems, arcs = self.decomposer_solver.decompositer(
            routes, centroids, angles, dist_matrix, vfreq, num_clients,
            sub_problem_size, record_path_size, seed
        )
        if subproblems is None and arcs is None:
            return None, None
        if arcs is None:
            return subproblems, arcs
        subproblems = self.arcs_to_paths(arcs)
        return subproblems, arcs

    def check_subproblems(self, subproblems):
        for sp in subproblems:
            if not sp:
                raise ValueError(f"Invalid subproblem format: {sp}")
            if isinstance(sp[0], list):
                raise ValueError(f"Invalid subproblem format: {sp}")

        node_to_sp = {}
        for sp_idx, nodes in enumerate(subproblems):
            for node in nodes:
                if node == 0:
                    continue
                if node < 1 or node > self.num_clients:
                    raise ValueError(
                        f"Invalid node id {node}, should be in [1, {self.num_clients}]"
                    )
                if node in node_to_sp:
                    raise ValueError(
                        f"Node {node} appears in multiple subproblems "
                        f"({node_to_sp[node]} and {sp_idx})"
                    )
                node_to_sp[node] = sp_idx

        covered = set(node_to_sp.keys())
        missing = [i for i in range(1, self.num_clients + 1) if i not in covered]
        if missing:
            raise Exception(f"Missing clients not assigned to any subproblem: {missing}")
        return True

    def arcs_to_paths(self, arcs):
        paths = []
        visited = [False for _ in range(self.num_clients + 1)]

        for arc in arcs:
            if visited[arc[0]] and visited[arc[1]]:
                continue

            is_back, back_idx = self.isArcInPathsBack(arc, paths)
            is_front, front_idx = self.isArcInPathsFront(arc, paths)

            if is_back or is_front:
                if is_back and is_front:
                    the_rest_len = self.sub_problem_size - len(paths[back_idx])
                    paths[back_idx] += paths[front_idx][:the_rest_len]
                    if the_rest_len >= len(paths[front_idx]):
                        del paths[front_idx]
                    else:
                        paths[front_idx] = paths[front_idx][the_rest_len:]
                    visited[arc[0]] = True
                    visited[arc[1]] = True
                elif is_back and len(paths) < self.sub_problem_size:
                    if not visited[arc[1]]:
                        paths[back_idx].append(arc[1])
                    visited[arc[1]] = True
                elif is_front and len(paths) < self.sub_problem_size:
                    if not visited[arc[0]]:
                        paths[front_idx].insert(0, arc[0])
                    visited[arc[0]] = True
                else:
                    if not visited[arc[0]] and not visited[arc[1]]:
                        paths.append(arc)
                        visited[arc[0]] = True
                        visited[arc[1]] = True
            else:
                if not visited[arc[0]] and not visited[arc[1]]:
                    paths.append(arc)
                    visited[arc[0]] = True
                    visited[arc[1]] = True

        rest_nodes = [i for i in range(1, self.num_clients + 1) if not visited[i]]
        for node in rest_nodes:
            paths.append([node])
        return [path for path in paths if len(path) > 0]

    @staticmethod
    def isArcInPathsBack(arc, paths):
        if paths is None:
            return False, -1
        for idx, path in enumerate(paths):
            if path[-1] == arc[0]:
                return True, idx
        return False, -1

    @staticmethod
    def isArcInPathsFront(arc, paths):
        if paths is None:
            return False, -1
        for idx, path in enumerate(paths):
            if path[0] == arc[1]:
                return True, idx
        return False, -1
