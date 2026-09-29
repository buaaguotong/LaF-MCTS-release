"""Final LaF-MCTS-generated component. Tier 3: recursive HGS sub-solver configuration.
The learned logic/parameters below are copied verbatim from the selected final solver.
"""

#
#An intensification-focused HGS with smaller populations, narrower granular neighbourhoods, fewer local-search operators, and aggressive penalty updates to quickly converge on high-quality feasible CVRP solutions under 300 customers.
def hgs_configuration():
  hgs_parameters = {
  'node_ops': ['Exchange11', 'Exchange21', 'Exchange22', 'SwapTails'],
  'route_ops': ['SwapStar'],
  'repair_probability': 0.5,
  'nb_iter_no_improvement': 2000,
  'min_pop_size': 20,
  'generation_size': 25,
  'nb_elites': 4,
  'nb_close': 8,
  'lb_diversity': 0.1,
  'ub_diversity': 0.4,
  'weight_wait_time': 0.0,
  'weight_time_warp': 0.0,
  'nb_granular': 25,
  'symmetric_proximity': True,
  'symmetric_neighbours': True,
  'repair_booster': 5,
  'solutions_between_updates': 50,
  'penalty_increase': 1.10,
  'penalty_decrease': 0.90,
  'target_feasible': 0.6
  }
  return hgs_parameters
