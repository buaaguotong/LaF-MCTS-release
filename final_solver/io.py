"""Small CVRPLIB/solution loader for the final solver release."""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np


def _header_value(lines, key, default=None):
    key = key.upper()
    for line in lines:
        if ':' in line:
            lhs, rhs = line.split(':', 1)
            if lhs.strip().upper() == key:
                return rhs.strip()
    return default


def read_vrp(path):
    path = Path(path)
    lines = [ln.strip() for ln in path.read_text().splitlines()]
    dim = int(_header_value(lines, 'DIMENSION'))
    capacity = int(float(_header_value(lines, 'CAPACITY')))
    ew_type = (_header_value(lines, 'EDGE_WEIGHT_TYPE', 'EUC_2D') or 'EUC_2D').upper()

    def section(name):
        try:
            start = lines.index(name) + 1
        except ValueError as exc:
            raise ValueError(f'Missing {name} in {path}') from exc
        out=[]
        for ln in lines[start:]:
            if not ln or ln == 'EOF' or ln.endswith('_SECTION'):
                break
            out.append(ln)
        return out

    coords = np.zeros((dim, 2), dtype=float)
    for ln in section('NODE_COORD_SECTION'):
        parts=ln.split()
        idx=int(parts[0]) - 1
        coords[idx]=[float(parts[1]), float(parts[2])]

    demands=np.zeros(dim, dtype=int)
    for ln in section('DEMAND_SECTION'):
        parts=ln.split()
        idx=int(parts[0]) - 1
        demands[idx]=int(float(parts[1]))

    diff=coords[:,None,:]-coords[None,:,:]
    dist=np.sqrt(np.sum(diff*diff, axis=-1))
    if ew_type in {'EUC_2D', 'EUC_2D_ROUNDED'}:
        # CVRPLIB/TSPLIB nearest-integer convention.
        dist=np.floor(dist + 0.5)
    elif ew_type not in {'EXACT_2D'}:
        raise NotImplementedError(
            f'EDGE_WEIGHT_TYPE={ew_type} is not supported by this lightweight loader. '
            'Use --instance-npz to provide the exact distance matrix used in your experiment.'
        )
    return {
        'filename': path.name,
        'coordinates': coords,
        'dist_matrix': dist.astype(float),
        'demands': demands,
        'capacity': capacity,
    }


def read_solution(path):
    path=Path(path)
    routes=[]
    cost=None
    for ln in path.read_text().splitlines():
        s=ln.strip()
        if not s:
            continue
        if s.lower().startswith('route'):
            # CVRPLIB solution format uses 1-based customer ids while the
            # solver internally uses depot=0 and customer ids 1..N.
            after=s.split(':',1)[1] if ':' in s else s
            route=[int(x) for x in re.findall(r'\d+', after)]
            routes.append(route)
        elif s.lower().startswith('cost'):
            nums=re.findall(r'[-+]?\d+(?:\.\d+)?',s)
            if nums:
                cost=float(nums[-1])
    return routes,cost


def read_npz(path):
    z=np.load(path, allow_pickle=False)
    required=['coordinates','dist_matrix','demands','capacity']
    missing=[k for k in required if k not in z]
    if missing:
        raise ValueError(f'Missing arrays in NPZ: {missing}')
    return {
        'filename': Path(path).name,
        'coordinates': np.asarray(z['coordinates']),
        'dist_matrix': np.asarray(z['dist_matrix']),
        'demands': np.asarray(z['demands']),
        'capacity': int(np.asarray(z['capacity']).item()),
        'num_vehicles': int(np.asarray(z['num_vehicles']).item()) if 'num_vehicles' in z else None,
        'opt_cost': float(np.asarray(z['opt_cost']).item()) if 'opt_cost' in z else None,
    }
