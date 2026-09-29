#!/usr/bin/env python3
"""Run the exact three LaF-MCTS-generated components as one HGS+decomposition solver."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from final_solver.io import read_npz, read_solution, read_vrp
from final_solver.generated import main_configuration as mc
from final_solver.generated import decomposition_strategy as de
from final_solver.generated import subsolver_configuration as sc


def build_parser():
    p=argparse.ArgumentParser(description='LaF-MCTS final decomposition-enhanced HGS solver')
    src=p.add_mutually_exclusive_group(required=True)
    src.add_argument('--instance', help='CVRPLIB .vrp file')
    src.add_argument('--instance-npz', help='NPZ with exact coordinates/dist_matrix/demands/capacity')
    p.add_argument('--solution', help='Optional CVRPLIB .sol file; used to obtain #vehicles and BKS')
    p.add_argument('--vehicles', type=int, help='Number of available vehicles (required if no .sol/NPZ value)')
    p.add_argument('--bks', type=float, help='Optional best-known objective for gap reporting')
    p.add_argument('--runtime', type=float, default=None,
                   help='Runtime in seconds. Default: N*2.4, where N is the number of customers.')
    p.add_argument('--output', default='solver_result.json', help='JSON output path')
    p.add_argument('--dry-run', action='store_true', help='Parse input and print learned components without importing PyVRP')
    return p


def main():
    args=build_parser().parse_args()
    data=read_npz(args.instance_npz) if args.instance_npz else read_vrp(args.instance)

    routes=[]
    sol_cost=None
    if args.solution:
        routes,sol_cost=read_solution(args.solution)

    num_vehicles=args.vehicles or data.get('num_vehicles') or (len(routes) if routes else None)
    if not num_vehicles:
        raise SystemExit('Please provide --vehicles or --solution (or num_vehicles in --instance-npz).')

    bks=args.bks
    if bks is None:
        bks=data.get('opt_cost') or sol_cost

    n_customers=len(data['demands'])-1
    runtime=args.runtime if args.runtime is not None else n_customers*2.4

    meta={
        'instance': data['filename'],
        'num_customers': n_customers,
        'num_vehicles': int(num_vehicles),
        'capacity': int(data['capacity']),
        'runtime_seconds': float(runtime),
        'bks': None if bks is None else float(bks),
        'main_configuration': mc.hgs_configuration(),
        'subsolver_configuration': sc.hgs_configuration(),
        'decomposition': 'final_solver.generated.decomposition_strategy.decompositer',
    }

    if args.dry_run:
        print(json.dumps(meta, indent=2))
        return

    # Import only here so --dry-run remains useful for checking a release on
    # machines where PyVRP is not yet installed.
    from final_solver.core.dehgs import DEHGS, get_sol_cost

    hgs=DEHGS(
        decomposer_solver=de,
        coordinates=data['coordinates'],
        dist_matrix=data['dist_matrix'],
        demands=data['demands'],
        capacity=int(data['capacity']),
        num_vehicle=int(num_vehicles),
        opt_cost=None if bks is None else float(bks),
        config=mc.hgs_configuration(),
        sub_config=sc.hgs_configuration(),
        macor=None,
    )
    result,_=hgs.solve(runtime=float(runtime))
    best=result.best
    best_routes=[]
    for route in best.routes():
        visits=list(route.visits())
        best_routes.append([0]+visits+[0])
    objective=round(get_sol_cost(data['dist_matrix'], best_routes))
    gap=None if bks is None else (objective/float(bks)-1.0)*100.0

    out={**meta, 'objective': objective, 'gap_percent': gap, 'routes': best_routes}
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps({'objective': objective, 'gap_percent': gap, 'output': args.output}, indent=2))


if __name__ == '__main__':
    main()
