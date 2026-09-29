# from __future__ import annotations
import sys
import types
import warnings

import numpy as np
import time
from typing import Callable, Collection, Tuple


from .decomposition import Decomposition

from pyvrp.ProgressPrinter import ProgressPrinter
from pyvrp.Result import Result
from pyvrp.Statistics import Statistics

from pyvrp.Population import Population
from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.search.SearchMethod import SearchMethod
from pyvrp.stop.StoppingCriterion import StoppingCriterion

from pyvrp import Client, Depot, VehicleType, Route
from pyvrp.GeneticAlgorithm import GeneticAlgorithmParams
from pyvrp.PenaltyManager import PenaltyParams, PenaltyManager
from pyvrp.Population import PopulationParams
from pyvrp.search import NeighbourhoodParams, LocalSearch, compute_neighbours
from pyvrp.search import NODE_OPERATORS
from pyvrp.search import ROUTE_OPERATORS

from pyvrp.stop import MaxIterations
from pyvrp.diversity import broken_pairs_distance
from pyvrp.crossover import selective_route_exchange, ordered_crossover

def get_sub_configuration(sub_configuration_module):
    sub_config = sub_configuration_module.hgs_configuration()

    return sub_config

def module_code(sub_configuration_code):
    try:
        # Suppress warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            if sub_configuration_code is not None:
                sub_configuration_module = types.ModuleType("sub_configuration")
                exec(sub_configuration_code, sub_configuration_module.__dict__)
                sys.modules[sub_configuration_module.__name__] = sub_configuration_module

                sub_config = get_sub_configuration(sub_configuration_module)
            else:
                sub_config = None
            return sub_config

    except Exception as e:
        print("Error code:", str(e))
        return np.inf



def get_sub_config(offspring):
    try:
        sub_configuration_code = offspring['code']
        sub_config = module_code(sub_configuration_code)
    except KeyError:
        print('key error ...')

        return None
    return sub_config



class MaxRuntime:
    """
    Criterion that stops after a specified maximum runtime (in seconds).
    """

    def __init__(self, max_runtime: float):
        if max_runtime < 0:
            raise ValueError("max_runtime < 0 not understood.")

        self._max_runtime = max_runtime
        self._start_runtime: float | None = None
        self._add = False

    def set_max_runtime(self, add_max_runtime: float):
        if not self._add:
            self._max_runtime = add_max_runtime + self._max_runtime
            self._add = True


    def __call__(self, best_cost: float) -> bool:
        if self._start_runtime is None:
            self._start_runtime = time.perf_counter()

        return time.perf_counter() - self._start_runtime > self._max_runtime


class GeneticAlgorithm:
    def __init__(
        self,
        decomposer_solver : types.ModuleType,
        data: ProblemData,
        pop_params: PopulationParams,
        gen_params : GeneticAlgorithmParams,
        penalty_params : PenaltyParams,
        neighbourhoodparams : NeighbourhoodParams,
        node_ops : list,
        route_ops : list,
        sub_params : dict,
        sub_config : dict,
        original_data : dict,
        recursive_id : int,
        seed : int = 2025,
        macor = None,
        display : bool = False,
        use_decomposition : bool = True,
    ):
        self._decomposer_solver = decomposer_solver
        self._data = data
        self._pop_params = pop_params
        self._gen_params = gen_params
        self._penalty_params = penalty_params
        self._neighbourhoodparams = neighbourhoodparams
        self._node_ops = node_ops
        self._route_ops = route_ops
        self._sub_params = sub_params
        self._sub_config = sub_config
        self._original_data = original_data
        self._seed = seed
        self._macor = macor

        self._llm_sc =False

        # subproblem parameters
        self._sub_pop_params = self._sub_params['pop_params']
        self._sub_gen_params = self._sub_params['gen_params']
        self._sub_penalty_params = self._sub_params['penalty_params']
        self._sub_neighbourhoodparams = self._sub_params['neighbourhoodparams']
        self._sub_node_ops = self._sub_params['node_ops']
        self._sub_route_ops = self._sub_params['route_ops']

        # main problem parameters
        self._pm = PenaltyManager.init_from(self._data, self._penalty_params)
        self._rng = RandomNumberGenerator(seed=self._seed)
        self._pop = Population(broken_pairs_distance, self._pop_params)
        self._neighbours = compute_neighbours(self._data, self._neighbourhoodparams)
        self._search = LocalSearch(self._data, self._rng, self._neighbours)
        for node_op in self._node_ops:
            self._search.add_node_operator(node_op(self._data))

        for route_op in self._route_ops:
            self._search.add_route_operator(route_op(self._data))
        self._crossover = selective_route_exchange if self._data.num_vehicles > 1 else ordered_crossover
        self._initial_solutions = [Solution.make_random(self._data, self._rng) for _ in
                                   range(self._pop_params.min_pop_size)]
        self._params = self._gen_params

        # Find best feasible initial solution if any exist, else set a random
        # infeasible solution (with infinite cost) as the initial best.
        self._best = min(self._initial_solutions, key=self._cost_evaluator.cost)

        self._config = original_data['config']
        self.recursive_id = recursive_id
        self._max_iterations = self._sub_config.get("max_iterations", 1000)

        self._sub_problem_size = self._config.get("sub_problem_size", 100)
        self._decomposition_iterations = self._config.get("decomposition_iterations", 200)
        self._record_path_size = self._config.get("record_path_size", 4)

        if self.recursive_id == 0:
            # self._sub_problem_size = self._config.get("sub_problem_size", 200)
            self._decomposition_iterations = self._config.get("decomposition_iterations", 200) * 10
            # self._record_path_size = self._config.get("record_path_size", 4)

        else:
            # self._sub_problem_size = self._sub_config.get("sub_problem_size", 200)
            self._decomposition_iterations = self._sub_config.get("decomposition_iterations", 200) * 10
            # self._record_path_size = self._sub_config.get("record_path_size", 4)

        self._decomposer = Decomposition(self._decomposer_solver, self._data, self._sub_problem_size, self._record_path_size, self._original_data, seed=self._seed)

        # Find best feasible initial solution if any exist, else set a random
        # infeasible solution (with infinite cost) as the initial best.


        # public
        self.num_clients = self._data.num_clients
        self.vfreq = np.zeros((self.num_clients + 1, self.num_clients + 1))
        self.best_routes = []
        self.use_decomposition = use_decomposition
        self.display = display


    @property
    def _cost_evaluator(self) -> CostEvaluator:
        return self._pm.cost_evaluator()

    def set_recursive_id(self, recursive_id):
        self.recursive_id = recursive_id
        # print(f'Recursive ID: {recursive_id}')

    #
    # def set_self_recursive_id(self, recursive_id):
    #     self.recursive_id = recursive_id

    def _change_sub_GA(self):
        sub_node_ops = self._sub_config.get("node_ops", None)
        sub_route_ops = self._sub_config.get("route_ops", None)

        if sub_node_ops == None:
            sub_node_ops = NODE_OPERATORS

        else:
            sub_node_ops = set_node_ops(sub_node_ops)

        if sub_route_ops == None:
            sub_route_ops = ROUTE_OPERATORS

        else:
            sub_route_ops = set_route_ops(sub_route_ops)

        self._sub_node_ops = sub_node_ops
        self._sub_route_ops = sub_route_ops

        # genetic
        self.sub_repair_probability = self._sub_config.get("repair_probability", 0.8)
        self.sub_nb_iter_no_improvement = self._sub_config.get("nb_iter_no_improvement", 20000)
        self.sub_decomposition_iterations = self._sub_config.get("decomposition_iterations", 10)

        # population-------------------
        self.sub_min_pop_size = self._sub_config.get("min_pop_size", 25)
        self.sub_generation_size = self._sub_config.get("generation_size", 40)
        self.sub_nb_elites = self._sub_config.get("nb_elites", 4)
        self.sub_nb_close = self._sub_config.get("nb_close", 5)
        self.sub_lb_diversity = self._sub_config.get("lb_diversity", 0.1)
        self.sub_ub_diversity = self._sub_config.get("ub_diversity", 0.5)

        # neigohourhood
        self.sub_weight_wait_time = self._sub_config.get("weight_wait_time", 0.2)
        self.sub_weight_time_warp = self._sub_config.get("weight_time_warp", 1.0)
        self.sub_nb_granular = self._sub_config.get("nb_granular", 40)
        self.sub_symmetric_proximity = self._sub_config.get("symmetric_proximity", True)
        self.sub_symmetric_neighbours = self._sub_config.get("symmetric_neighbours", False)

        # penalty
        self.sub_repair_booster = self._sub_config.get("repair_booster", 12)
        self.sub_solutions_between_updates = self._sub_config.get("solutions_between_updates", 50)
        self.sub_penalty_increase = self._sub_config.get("penalty_increase", 1.34)
        self.sub_penalty_decrease = self._sub_config.get("penalty_decrease", 0.32)
        self.sub_target_feasible = self._sub_config.get("target_feasible", 0.43)

        # HGS
        self._sub_pop_params = PopulationParams(
            min_pop_size=self.sub_min_pop_size,  # 最小的种群大小
            generation_size=self.sub_generation_size,  # 插入新种群的数量
            nb_elite=self.sub_nb_elites,  # 精英个体数量
            nb_close=self.sub_nb_close,  # 相似solution的数量
            lb_diversity=self.sub_lb_diversity,  # 多样性下限
            ub_diversity=self.sub_ub_diversity,  # 多样性上限
        )

        self._sub_gen_params = GeneticAlgorithmParams(
            repair_probability=self.sub_repair_probability,
            nb_iter_no_improvement=self.sub_nb_iter_no_improvement,
        )

        self._sub_penalty_params = PenaltyParams(
            repair_booster=self.sub_repair_booster,
            solutions_between_updates=self.sub_solutions_between_updates,
            penalty_increase=self.sub_penalty_increase,
            penalty_decrease=self.sub_penalty_decrease,
            target_feasible=self.sub_target_feasible,
        )

        self._sub_neighbourhoodparams = NeighbourhoodParams(
            weight_wait_time=self.sub_weight_wait_time,
            weight_time_warp=self.sub_weight_time_warp,
            nb_granular=self.sub_nb_granular,
            symmetric_proximity=self.sub_symmetric_proximity,
            symmetric_neighbours=self.sub_symmetric_neighbours,
        )

        self._sub_params = {
            'pop_params': self._sub_pop_params,
            'gen_params': self._sub_gen_params,
            'penalty_params': self._sub_penalty_params,
            'neighbourhoodparams': self._sub_neighbourhoodparams,
            'node_ops': self._sub_node_ops,
            'route_ops': self._sub_route_ops,
        }


    def run(
        self,
        stop: StoppingCriterion,
        collect_stats: bool = False,
        display: bool = False,
    ):
        sub_offsprings = None
        print_progress = ProgressPrinter(should_print=display) #输出训练信息
        print_progress.start(self._data)

        start = time.perf_counter()
        stats = Statistics(collect_stats=collect_stats)
        iters = 0
        # iters_no_improvement = 1
        iters_no_improvement = 0


        for sol in self._initial_solutions:
            self._pop.add(sol, self._cost_evaluator)


        while not stop(self._cost_evaluator.cost(self._best)):
            iters += 1

            if iters_no_improvement == self._params.nb_iter_no_improvement:
                print_progress.restart()

                # iters_no_improvement = 1
                iters_no_improvement = 0
                self._pop.clear()

                for sol in self._initial_solutions:
                    self._pop.add(sol, self._cost_evaluator)

            curr_best = self._cost_evaluator.cost(self._best)

            parents = self._pop.select(self._rng, self._cost_evaluator)
            offspring = self._crossover(
                parents, self._data, self._cost_evaluator, self._rng
            )
            self._improve_offspring(offspring)

            routes = []
            for route in self._best.routes():
                routes.append(route.visits())

            self.best_routes = routes
            self._updata_vfreq()

            new_best = self._cost_evaluator.cost(self._best)

            if new_best < curr_best:
                # iters_no_improvement = 1
                iters_no_improvement = 0
            else:
                iters_no_improvement += 1

            # decomposition
            if self.use_decomposition and iters % self._decomposition_iterations == 0 and self.recursive_id < 1:
                new_sol = self._best
                np.random.seed(2025)
                elites = self._pop.select(self._rng, self._cost_evaluator, k=self._pop._params.nb_elite)
                elite = np.random.choice(elites) #Solution

                subproblems, arcs = self._decomposer.decompose(elite, self.vfreq)

                if subproblems is not None and len(subproblems) > 1:

                    # if don't decompose the problem, the rest iterations don't go to recursive decomposition module
                    routers = []

                    if arcs is None:
                        # get sub configuration
                        if self.recursive_id == 0 and not self._llm_sc and self._macor is not None:
                            # get the subproblem characterizations
                            sc_start_time = time.time()
                            sub_offsprings = self._macor.initialization_sc(subproblems, self._original_data)
                            self._llm_sc = True
                            self._sub_config = get_sub_config(sub_offsprings)
                            self._change_sub_GA()

                            sc_end_time = time.time()
                            stop.set_max_runtime(abs(sc_end_time - sc_start_time))

                        sub_instances, subproblem_indices = self._init_route_instance(subproblems)
                        de_gas = []
                        sub_solutions = []
                        for sub_instance in sub_instances:
                            de_ga = GeneticAlgorithm(
                                decomposer_solver=self._decomposer_solver,
                                data=sub_instance,
                                pop_params=self._sub_pop_params,
                                gen_params=self._sub_gen_params,
                                penalty_params=self._sub_penalty_params,
                                neighbourhoodparams=self._sub_neighbourhoodparams,
                                node_ops=self._sub_node_ops,
                                route_ops=self._sub_route_ops,
                                sub_params=self._sub_params,
                                sub_config=self._sub_config,
                                original_data=self._original_data,
                                recursive_id=self.recursive_id + 1,
                                seed=self._seed,
                                macor=self._macor,
                                display=self.display,
                                use_decomposition=self.use_decomposition,
                            )
                            # de_ga.set_recursive_id(self.recursive_id + 1)
                            de_gas.append(de_ga)

                        for idx, de_ga in enumerate(de_gas):
                            res, _ = de_ga.run(stop=MaxIterations(self._max_iterations), display=display)  # Result
                            sub_indices = subproblem_indices[idx]

                            sub_best = res.best # Solution
                            sub_routes = [route.visits() for route in sub_best.routes()] #List[Route] -> list[int] except depot

                            for sub_route in sub_routes:
                                sub_solutions.append(sub_indices[np.array(sub_route)].tolist()) # list[int]


                        for solution in sub_solutions:
                            routers.append(Route(self._data, visits=solution, vehicle_type=0)) # one vehicle_type

                        new_sol = Solution(self._data, routes=routers)
                        self._improve_offspring(new_sol)

                    else:
                        sub_instances, subproblem_indices, super_problems, sub_data = self._init_path_instance(subproblems, arcs)
                        if self.recursive_id == 0 and len(sub_instances) > 1:
                            # get sub configuration
                            if self.recursive_id == 0 and not self._llm_sc and self._macor is not None:
                                # get the subproblem characterizations
                                sc_start_time = time.time()
                                sub_offsprings = self._macor.initialization_sc(super_problems, sub_data)
                                self._llm_sc = True
                                self._sub_config = get_sub_config(sub_offsprings)
                                self._change_sub_GA()

                                sc_end_time = time.time()

                                stop.set_max_runtime(abs(sc_end_time - sc_start_time))

                            de_gas = []
                            sub_solutions = []
                            for sub_instance in sub_instances:
                                de_ga = GeneticAlgorithm(
                                    decomposer_solver=self._decomposer_solver,
                                    data=sub_instance,
                                    pop_params=self._sub_pop_params,
                                    gen_params=self._sub_gen_params,
                                    penalty_params=self._sub_penalty_params,
                                    neighbourhoodparams=self._sub_neighbourhoodparams,
                                    node_ops=self._sub_node_ops,
                                    route_ops=self._sub_route_ops,
                                    sub_params=self._sub_params,
                                    sub_config=self._sub_config,
                                    original_data=self._original_data,
                                    recursive_id=self.recursive_id + 1,
                                    seed=self._seed,
                                    macor=self._macor,
                                    display=self.display,
                                    use_decomposition=self.use_decomposition,
                                )
                                # de_ga.set_recursive_id(self.recursive_id + 1)
                                de_gas.append(de_ga)

                            for idx, de_ga in enumerate(de_gas):
                                res, _ = de_ga.run(stop=MaxIterations(self._max_iterations), display=display)  # Result
                                sub_indices = subproblem_indices[idx]

                                sub_best = res.best  # Solution
                                sub_routes = [route.visits() for route in
                                              sub_best.routes()]  # List[Route] -> list[int] except depot

                                for sub_route in sub_routes:
                                    sub_solutions.append(sub_indices[np.array(sub_route)].tolist())  # list[int]

                            sub_solutions = self._original_solution_from_paths(sub_solutions, subproblems)

                            for solution in sub_solutions:
                                routers.append(Route(self._data, visits=solution, vehicle_type=0))  # one vehicle_type

                            new_sol = Solution(self._data, routes=routers)
                            self._improve_offspring(new_sol)

                        else:
                            if self.recursive_id > 0:
                                end = time.perf_counter() - start
                                res = Result(self._best, stats, iters, end)
                                # new_sol = res.best
                                # # return res

                else:
                    if self.recursive_id > 0:
                        end = time.perf_counter() - start
                        res = Result(self._best, stats, iters, end)
                        # new_sol = res.best
                        # # return res
                    # print('子问题只有一个')
                    new_sol = self._best

                    # new_sol = self._best
                    # self.use_decomposition = False

                if self.display and iters % self._decomposition_iterations == 0:
                    print(
                        "Recursive_id : {} | "
                        "Best cost : {} | "
                        "Feasible solutions : {} | "
                        "Infeasible solutions : {} | "
                        "Iterations: {} | "
                        "No improvement : {} | "
                        "Decomposer cost : {}".format(
                            self.recursive_id,
                            self._best.distance_cost(),
                            self._pop.num_feasible(),
                            self._pop.num_infeasible(),
                            iters,
                            iters_no_improvement,
                            new_sol.distance_cost()))


            elif self.display and iters % self._decomposition_iterations == 0:
                print(
                    "Recursive_id : {} | "
                    "Best cost : {} | "
                    "Feasible solutions : {} | "
                    "Infeasible solutions : {} | "
                    "Iterations: {} | "
                    "No improvement : {} | "
                    "Decomposer cost : {}".format(
                        self.recursive_id,
                        self._best.distance_cost(),
                        self._pop.num_feasible(),
                        self._pop.num_infeasible(),
                        iters,
                        iters_no_improvement,
                        np.inf))

            stats.collect_from(self._pop, self._cost_evaluator)
            print_progress.iteration(stats)

        end = time.perf_counter() - start
        res = Result(self._best, stats, iters, end)

        print_progress.end(res)

        return res, sub_offsprings

    def _improve_offspring(self, sol: Solution):

        def is_new_best(sol):
            cost = self._cost_evaluator.cost(sol)
            best_cost = self._cost_evaluator.cost(self._best)
            return cost < best_cost

        # if is_new_best(sol):
        #     self._best = sol

        sol = self._search(sol, self._cost_evaluator)
        self._pop.add(sol, self._cost_evaluator)
        self._pm.register(sol)

        if is_new_best(sol):
            self._best = sol

        # Possibly repair if current solution is infeasible. In that case, we
        # penalise infeasibility more using a penalty booster.
        if (
            not sol.is_feasible()
            and self._rng.rand() < self._params.repair_probability
        ):
            sol = self._search(sol, self._pm.booster_cost_evaluator())

            if sol.is_feasible():
                self._pop.add(sol, self._cost_evaluator)
                self._pm.register(sol)

            if is_new_best(sol):
                self._best = sol

    def _updata_vfreq(self):
        for route in self.best_routes:
            for a in range(len(route)):
                for b in range(a + 1, len(route)):
                    i, j = route[a], route[b]
                    self.vfreq[i, j] += 1
                    self.vfreq[j, i] += 1

    def _display(self, routes: list[list], subproblems: list[list]) -> None:
        for i, route in enumerate(routes):
            print(f"Idx: {i} : Route: {route}")
        print('==' * 100)
        print('==' * 100)
        for i, subproblem in enumerate(subproblems):
            print(f"Idx: {i} : Route: {subproblem}")

    def _init_route_instance(self, subproblems: list[list]) -> Tuple[list[ProblemData],list[np.ndarray]]:
        # initialization instance
        coordinates = self._original_data['coordinates']
        dist_matrix = self._original_data['dist_matrix']
        demands = self._original_data['demands']
        capacity = self._original_data['capacity']
        num_vehicle = self._original_data['num_vehicle']

        depot = [Depot(x=coordinates[0][0], y=coordinates[0][1])]
        vehicle_types = [VehicleType(num_available=num_vehicle, capacity=[capacity], start_depot=0, end_depot=0)]

        instances = []
        indices = []
        for subproblem in subproblems:
            clients = [Client(x=coordinates[idx][0], y=coordinates[idx][1], delivery=[demands[idx]]) for idx in subproblem]

            subproblem_indices = np.array([0] + subproblem)
            indices.append(subproblem_indices)
            sub_distance_matrix = dist_matrix[np.ix_(subproblem_indices, subproblem_indices)]

            sub_instance = ProblemData(depots=depot, clients=clients, vehicle_types=vehicle_types,
                                        distance_matrices=[sub_distance_matrix], duration_matrices=[sub_distance_matrix])

            instances.append(sub_instance)

        return instances, indices

    def _init_path_instance(self, subproblems: list[list], arcs : list[list]) -> Tuple[list[ProblemData],list[np.ndarray]]:
        # initialization instance
        coordinates = self._original_data['coordinates']
        dist_matrix = self._original_data['dist_matrix']
        demands = self._original_data['demands']
        capacity = self._original_data['capacity']
        num_vehicle = self._original_data['num_vehicle']
        depot = [Depot(x=coordinates[0][0], y=coordinates[0][1])]
        vehicle_types = [VehicleType(num_available=num_vehicle, capacity=[capacity], start_depot=0, end_depot=0)]

        # construct the path clients

        sub_coordinates = self._get_subcoordinates(subproblems, coordinates)
        sub_demands = self._get_subdemands(subproblems, demands)
        sub_dist_matrix = self._get_sub_dist_matrix(subproblems, dist_matrix)
        instances = []
        indices = []
        temp_problems = [i for i in range(1, len(subproblems) + 1)]
        super_problems = []

        for i in range(int(np.ceil(len(subproblems) / self._sub_problem_size))):
            super_problems.append(temp_problems[i * self._sub_problem_size: (i + 1) * self._sub_problem_size])

        for subproblem in super_problems:
            clients = [Client(x=sub_coordinates[idx][0], y=sub_coordinates[idx][1], delivery=[sub_demands[idx]]) for idx in subproblem]
            subproblem_indices = np.array([0] + subproblem)
            indices.append(subproblem_indices)
            sub_distance_matrix = sub_dist_matrix[np.ix_(subproblem_indices, subproblem_indices)]

            sub_instance = ProblemData(depots=depot, clients=clients, vehicle_types=vehicle_types,
                                        distance_matrices=[sub_distance_matrix], duration_matrices=[sub_distance_matrix])

            instances.append(sub_instance)

        sub_data = {
            'coordinates': sub_coordinates,
            'demands': sub_demands,
            'capacity': capacity,
        }
        return instances, indices, super_problems, sub_data

    def _original_solution_from_paths(self, sub_solutions, subproblems):

        original_solutions = []
        for sol in sub_solutions:
            temp_solution = []
            for node in sol:
                temp_solution += subproblems[node - 1]

            original_solutions.append(temp_solution)

        return original_solutions

    def _get_subcoordinates(self, subproblems, coordinates):

        # sub_coordinates = np.zeros((len(subproblems) + 1, len(subproblems) + 1))
        sub_coordinates = np.zeros((len(subproblems) + 1, 2), dtype=int)
        sub_coordinates[0, 0] = coordinates[0, 0]
        sub_coordinates[0, 1] = coordinates[0, 1]
        for idx, subproblem in enumerate(subproblems):
            sub_coordinates[idx + 1, 0] = coordinates[subproblem[0], 0]
            sub_coordinates[idx + 1, 1] = coordinates[subproblem[0], 1]


        return sub_coordinates

    def _get_subdemands(self, subproblems, demands):

        sub_demands = np.zeros(len(subproblems) + 1, dtype=int)

        for idx, subproblem in enumerate(subproblems):

            sub_demands[idx + 1] = sum(demands[node] for node in subproblem)


        return sub_demands

    def _get_sub_dist_matrix(self, subproblems, dist_matrix):

        sub_dist_matrix = np.zeros((len(subproblems) + 1, len(subproblems) + 1))
        for idx, subproblem_x in enumerate(subproblems):
            sub_dist_matrix[idx + 1, 0] = dist_matrix[subproblem_x[0], 0]
            sub_dist_matrix[0, idx + 1] = dist_matrix[subproblem_x[0], 0]
            for idy, subproblem_y in enumerate(subproblems):
                if idx == idy:
                    continue
                sub_dist_matrix[idx + 1, idy + 1] = dist_matrix[subproblem_x[-1], subproblem_y[0]]

        return sub_dist_matrix

def set_node_ops(node_ops):
    node_ops_indix = {
    'Exchange10' : 0,
    'Exchange20': 1,
    'Exchange30' : 2,
    'Exchange11': 3,
    'Exchange21' : 4,
    'Exchange31' : 5,
    'Exchange22' : 6,
    'Exchange32' : 7,
    'Exchange33' : 8,
    'SwapTails' : 9,
    'TripRelocate' : 10
    }
    select_ops = []
    for op in node_ops:
        select_ops.append(NODE_OPERATORS[node_ops_indix[op]])

    return select_ops

def set_route_ops(route_ops):
    route_ops_indix = {
    'SwapRoutes' : 0,
    'SwapStar' : 1
    }
    select_ops = []
    for op in route_ops:
        select_ops.append(ROUTE_OPERATORS[route_ops_indix[op]])

    return select_ops


class DEHGS:
    def __init__(self,
                 decomposer_solver : types.ModuleType,
                 coordinates : np.ndarray,
                 dist_matrix : np.ndarray,
                 demands : np.ndarray,
                 capacity : int,
                 num_vehicle : int,
                 opt_cost : int,
                 config: dict,
                 sub_config : dict = None,
                 macor = None,
                 ):
        #original data
        self.macor = macor
        self.decomposer_solver = decomposer_solver
        self.display = True
        if sub_config is None:
            self.sub_config = config
        else:
            self.sub_config = sub_config
        self.original_data = {
            'coordinates' : coordinates,
            'dist_matrix' : dist_matrix,
            'demands' : demands,
            'capacity' : capacity,
            'num_vehicle' : num_vehicle,
            'config' : config
        }
        self.seed = 2025
        self.coordinates = coordinates
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.capacity = capacity
        self.num_vehicle = num_vehicle
        self.opt_cost = opt_cost

        #initialization instance
        self.depot = [Depot(x=coordinates[0][0], y=coordinates[0][1])]
        self.clients = [Client(x=coordinates[idx][0], y=coordinates[idx][1], delivery=[demands[idx]]) for idx in
                   range(1, len(demands))]
        self.vehicle_types = [VehicleType(num_available=num_vehicle, capacity=[capacity], start_depot=0, end_depot=0)]
        self.instance = ProblemData(depots=self.depot, clients=self.clients, vehicle_types=self.vehicle_types,
                               distance_matrices=[dist_matrix], duration_matrices=[dist_matrix])


        # config 外壳HGS的管理

        node_ops = config.get("node_ops", None)
        route_ops = config.get("route_ops", None)

        if node_ops == None:
            node_ops = NODE_OPERATORS

        else:
            node_ops = set_node_ops(node_ops)

        if route_ops == None:
            route_ops = ROUTE_OPERATORS

        else:
            route_ops = set_route_ops(route_ops)

        self.node_ops = node_ops
        self.route_ops = route_ops

        #genetic
        self.repair_probability = config.get("repair_probability", 0.8)
        self.nb_iter_no_improvement = config.get("nb_iter_no_improvement", 20000)
        self.decomposition_iterations = config.get("decomposition_iterations", 10)

        #population-------------------
        self.min_pop_size = config.get("min_pop_size", 25)
        self.generation_size = config.get("generation_size", 40)
        self.nb_elites = config.get("nb_elites", 4)
        self.nb_close = config.get("nb_close", 5)
        self.lb_diversity = config.get("lb_diversity", 0.1)
        self.ub_diversity = config.get("ub_diversity", 0.5)

        #neigohourhood
        self.weight_wait_time = config.get("weight_wait_time", 0.2)
        self.weight_time_warp = config.get("weight_time_warp", 1.0)
        self.nb_granular = config.get("nb_granular", 40)
        self.symmetric_proximity = config.get("symmetric_proximity", True)
        self.symmetric_neighbours = config.get("symmetric_neighbours", False)

        #penalty
        self.repair_booster = config.get("repair_booster", 12)
        self.solutions_between_updates = config.get("solutions_between_updates", 50)
        self.penalty_increase = config.get("penalty_increase", 1.34)
        self.penalty_decrease = config.get("penalty_decrease", 0.32)
        self.target_feasible = config.get("target_feasible", 0.43)

        #HGS
        self.pop_params = PopulationParams(
            min_pop_size=self.min_pop_size, #最小的种群大小
            generation_size=self.generation_size, #插入新种群的数量
            nb_elite=self.nb_elites, #精英个体数量
            nb_close=self.nb_close, #相似solution的数量
            lb_diversity=self.lb_diversity, #多样性下限
            ub_diversity=self.ub_diversity, #多样性上限
        )

        self.gen_params = GeneticAlgorithmParams(
            repair_probability=self.repair_probability,
            nb_iter_no_improvement=self.nb_iter_no_improvement,
        )

        self.penalty_params = PenaltyParams(
            repair_booster=self.repair_booster,
            solutions_between_updates=self.solutions_between_updates,
            penalty_increase=self.penalty_increase,
            penalty_decrease=self.penalty_decrease,
            target_feasible=self.target_feasible,
        )

        self.neighbourhoodparams = NeighbourhoodParams(
            weight_wait_time=self.weight_wait_time,
            weight_time_warp=self.weight_time_warp,
            nb_granular=self.nb_granular,
            symmetric_proximity=self.symmetric_proximity,
            symmetric_neighbours=self.symmetric_neighbours,
        )

        self.max_iterations = config.get("max_iterations", 1000)
        self.sub_problem_size = config.get("sub_problem_size", 50)

        self.ga_iterations = 5
        self.penalty_factor = 10
        self.num_init = 30
        self.early_stop_count = 100
        self.early_stop = False
        self.best_solution = None

        self.min_penalty = 10
        self.max_penalty = 100000.0
        self.load_penalty = 0.3 * self.max_penalty

        self.no_improvement_count = 0
        self.max_no_improvement = 10
        self.max_pop_size = self.pop_params.max_pop_size

        self.cvrp_costevaluator = CostEvaluator([self.load_penalty], 0.0, 0.0)

        self._initialize_sub_GA()

        self._initialize_GA()

    def _initialize_sub_GA(self):
        sub_node_ops = self.sub_config.get("node_ops", None)
        sub_route_ops = self.sub_config.get("route_ops", None)

        if sub_node_ops == None:
            sub_node_ops = NODE_OPERATORS

        else:
            sub_node_ops = set_node_ops(sub_node_ops)

        if sub_route_ops == None:
            sub_route_ops = ROUTE_OPERATORS

        else:
            sub_route_ops = set_route_ops(sub_route_ops)

        self.sub_node_ops = sub_node_ops
        self.sub_route_ops = sub_route_ops

        # genetic
        self.sub_repair_probability = self.sub_config.get("repair_probability", 0.8)
        self.sub_nb_iter_no_improvement = self.sub_config.get("nb_iter_no_improvement", 20000)
        self.sub_decomposition_iterations = self.sub_config.get("decomposition_iterations", 10)

        # population-------------------
        self.sub_min_pop_size = self.sub_config.get("min_pop_size", 25)
        self.sub_generation_size = self.sub_config.get("generation_size", 40)
        self.sub_nb_elites = self.sub_config.get("nb_elites", 4)
        self.sub_nb_close = self.sub_config.get("nb_close", 5)
        self.sub_lb_diversity = self.sub_config.get("lb_diversity", 0.1)
        self.sub_ub_diversity = self.sub_config.get("ub_diversity", 0.5)

        # neigohourhood
        self.sub_weight_wait_time = self.sub_config.get("weight_wait_time", 0.2)
        self.sub_weight_time_warp = self.sub_config.get("weight_time_warp", 1.0)
        self.sub_nb_granular = self.sub_config.get("nb_granular", 40)
        self.sub_symmetric_proximity = self.sub_config.get("symmetric_proximity", True)
        self.sub_symmetric_neighbours = self.sub_config.get("symmetric_neighbours", False)

        # penalty
        self.sub_repair_booster = self.sub_config.get("repair_booster", 12)
        self.sub_solutions_between_updates = self.sub_config.get("solutions_between_updates", 50)
        self.sub_penalty_increase = self.sub_config.get("penalty_increase", 1.34)
        self.sub_penalty_decrease = self.sub_config.get("penalty_decrease", 0.32)
        self.sub_target_feasible = self.sub_config.get("target_feasible", 0.43)

        # HGS
        self.sub_pop_params = PopulationParams(
            min_pop_size=self.sub_min_pop_size,  # 最小的种群大小
            generation_size=self.sub_generation_size,  # 插入新种群的数量
            nb_elite=self.sub_nb_elites,  # 精英个体数量
            nb_close=self.sub_nb_close,  # 相似solution的数量
            lb_diversity=self.sub_lb_diversity,  # 多样性下限
            ub_diversity=self.sub_ub_diversity,  # 多样性上限
        )

        self.sub_gen_params = GeneticAlgorithmParams(
            repair_probability=self.sub_repair_probability,
            nb_iter_no_improvement=self.sub_nb_iter_no_improvement,
        )

        self.sub_penalty_params = PenaltyParams(
            repair_booster=self.sub_repair_booster,
            solutions_between_updates=self.sub_solutions_between_updates,
            penalty_increase=self.sub_penalty_increase,
            penalty_decrease=self.sub_penalty_decrease,
            target_feasible=self.sub_target_feasible,
        )

        self.sub_neighbourhoodparams = NeighbourhoodParams(
            weight_wait_time=self.sub_weight_wait_time,
            weight_time_warp=self.sub_weight_time_warp,
            nb_granular=self.sub_nb_granular,
            symmetric_proximity=self.sub_symmetric_proximity,
            symmetric_neighbours=self.sub_symmetric_neighbours,
        )

        self.sub_params = {
            'pop_params': self.sub_pop_params,
            'gen_params': self.sub_gen_params,
            'penalty_params': self.sub_penalty_params,
            'neighbourhoodparams': self.sub_neighbourhoodparams,
            'node_ops': self.sub_node_ops,
            'route_ops': self.sub_route_ops,
        }

    def _initialize_GA(self):
        self.algo = GeneticAlgorithm(
            decomposer_solver = self.decomposer_solver,
            data=self.instance,
            pop_params = self.pop_params,
            gen_params = self.gen_params,
            penalty_params = self.penalty_params,
            neighbourhoodparams = self.neighbourhoodparams,
            node_ops = self.node_ops,
            route_ops = self.route_ops,
            sub_params = self.sub_params,
            sub_config = self.sub_config,
            original_data = self.original_data,
            recursive_id=0,
            seed = self.seed,
            macor= self.macor,
        )

        # self.algo.set_recursive_id(0)

    def solve(self, runtime = 120):

        # result = self.algo.run(stop=MaxIterations(self.max_iterations), display=False)
        # result = self.algo.run(stop=MaxIterations(1000), display=True)
        # start_time = time.time()
        result, suboffspring = self.algo.run(stop=MaxRuntime(runtime), display=False)
        # end_time = time.time()

        # temp_time = end_time - start_time
        return result, suboffspring

def check_subproblems(subproblems,num_clients):
    for sp in subproblems:
        if not sp:
            raise ValueError(f"Invalid subproblem format: {sp}")
        if isinstance(sp[0], list):
            raise ValueError(f"Invalid subproblem format: {sp}")


    node_to_sp = {}
    for sp_idx, nodes in enumerate(subproblems):
        for node in nodes:
            if node == 0:  # depot ignore
                continue
            if node < 1 or node > num_clients:
                raise ValueError(f"Invalid node id {node}, should be in [1, {num_clients}]")
            if node in node_to_sp:
                raise ValueError(f"Node {node} appears in multiple subproblems "
                                 f"({node_to_sp[node]} and {sp_idx})")
            node_to_sp[node] = sp_idx

    covered = set(node_to_sp.keys())

    missing = [i for i in range(1, num_clients + 1) if i not in covered]
    if missing:
        raise Exception(f"⚠️ Missing clients not assigned to any subproblem: {missing}")

    return True

def get_router_cost(distance_matrix ,route):

    cost = 0

    for i in range(len(route) - 1):
        cost += distance_matrix[route[i], route[i+1]]

    return cost

def get_sol_cost(distance_matrix ,solutions):

    cost = 0

    for route in solutions:
        cost += get_router_cost(distance_matrix, route)

    return cost

def hgs_solver(decomposer_solver, coordinates: np.ndarray, dist_matrix: np.ndarray, demands: np.ndarray, capacity: int,
               num_vehicle: int, opt_cost : int, config : dict, sub_config : dict = None) -> float:

    # initialization the instance

    hgs = DEHGS(decomposer_solver, coordinates, dist_matrix, demands, capacity, num_vehicle, opt_cost, config, sub_config)

    result = hgs.solve()

    cost = result.cost()

    print(f'HGS得到的cost是: {cost}')

    gap = (cost / opt_cost - 1) * 100.0

    print(f'HGS得到的gap是: {gap:2f} %')

    return gap

def hgs_solver_decomposer(decomposer_solver,coordinates: np.ndarray, dist_matrix: np.ndarray, demands: np.ndarray, capacity: int,
               num_vehicle: int, opt_cost : int, config : dict, sub_config : dict = None, runtime : int = 120, macor = None) -> float:

    # initialization the instance

    hgs = DEHGS(decomposer_solver, coordinates, dist_matrix, demands, capacity, num_vehicle, opt_cost, config, sub_config, macor)

    result, suboffspring = hgs.solve(runtime)

    best = result.best

    test_routes = []
    for route in best.routes():
        route = route.visits()

        route = [0] + route + [0]

        test_routes.append(route)

    cost = get_sol_cost(dist_matrix, test_routes)

    cost = round(cost)

    # cost = result.cost()

    print(f'HGS得到的cost是: {cost}')

    if opt_cost is not None:
        gap = (cost / opt_cost - 1) * 100.0
        print(f'HGS得到的gap是: {gap:2f} %')
    else:
        gap = np.inf

    return gap, cost, suboffspring

def solve_cvrp(idx,num_vehicles, coordinates,dist_matrix, demands, routes, opt_cost, capacity, configuration_module
               , sub_configuration_module, decomposition_module, runtime, macor = None):

    try:
        if configuration_module is None:
            config = {}
        else:
            config = configuration_module.hgs_configuration()

        if sub_configuration_module is None:
            sub_config = {}
        else:
            sub_config = sub_configuration_module.hgs_configuration()

        gap, objection, suboffspring = hgs_solver_decomposer(decomposition_module, coordinates, dist_matrix, demands, capacity, num_vehicles, opt_cost, config, sub_config, runtime, macor)

    except ValueError as e:

        print(f'code error : {e}')
        raise RuntimeError(e)

    return objection, suboffspring

if __name__ == '__main__':
    pass
