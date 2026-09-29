"""Final LaF-MCTS-generated component. Tier 1: overall HGS framework/configuration.
The learned logic/parameters below are copied verbatim from the selected final solver.
"""

#
#A diversified CVRP HGS configuration using broader exchange operators and a moderately frequent decomposition with moderate subproblem and path sizes to balance intensification and main-problem sensitivity.
def hgs_configuration():
  hgs_parameters = {
    'node_ops': ['Exchange10', 'Exchange20', 'Exchange30', 'Exchange11', 'Exchange21', 'Exchange22', 'SwapTails', 'TripRelocate'],
    'route_ops': ['SwapRoutes', 'SwapStar'],
    'repair_probability': 0.72,
    'nb_iter_no_improvement': 14000,
    'min_pop_size': 22,
    'generation_size': 45,
    'nb_elites': 4,
    'nb_close': 9,
    'lb_diversity': 0.12,
    'ub_diversity': 0.38,
    'weight_wait_time': 0.0,
    'weight_time_warp': 0.0,
    'nb_granular': 13,
    'symmetric_proximity': True,
    'symmetric_neighbours': True,
    'repair_booster': 7,
    'solutions_between_updates': 300,
    'penalty_increase': 1.18,
    'penalty_decrease': 0.82,
    'target_feasible': 0.38,
    'decomposition_iterations': 8000,
    'sub_problem_size': 95,
    'record_path_size': 5
  }
  return hgs_parameters
