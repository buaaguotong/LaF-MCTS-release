"""Final LaF-MCTS-generated component. Tier 2: decomposition strategy.
The learned logic/parameters below are copied verbatim from the selected final solver.
"""

#
#Arc-based frequent directed arc mining: add depot to each elite route, count consecutive arcs, score each arc by frequency visit frequency and inverse distance, and return the top capacity-scaled arcs while subproblems is None.
import random
import numpy as np

def decompositer(routes, centroids, angles, dist_matrix, vfreq, num_clients, sub_problem_size, record_path_size, seed):
    subproblems = None
    arcs = []
    random.seed(seed)
    np.random.seed(seed)
    try:
        n = int(num_clients)
    except Exception:
        n = 0
    if n <= 0:
        return subproblems, arcs
    cap = int(sub_problem_size) if sub_problem_size else n
    if cap < 1:
        cap = 1
    if cap > n:
        cap = n
    if routes is None:
        routes = []
    count = {}
    for r in routes:
        if not r:
            continue
        seq = [0]
        for c in r:
            try:
                ci = int(c)
            except Exception:
                continue
            if 1 <= ci <= n:
                seq.append(ci)
        if len(seq) <= 1:
            continue
        seq.append(0)
        L = len(seq)
        for i in range(L - 1):
            u = seq[i]
            v = seq[i + 1]
            if u == v:
                continue
            key = (u, v)
            count[key] = count.get(key, 0) + 1
    if not count:
        return subproblems, arcs
    scored = []
    for (u, v), cnt in count.items():
        try:
            d = float(dist_matrix[u][v])
            if d < 0:
                d = -d
            if d == 0:
                d = 1e-9
        except Exception:
            d = 1.0
        try:
            f = float(vfreq[u][v])
            if f < 0:
                f = -f
        except Exception:
            f = 1.0
        score = (1.0 + float(cnt)) * (1.0 + f) / (1.0 + d)
        if u == 0 or v == 0:
            score *= 1.2
        scored.append((score, cnt, u, v))
    scored.sort(key=lambda x: (-x[0], -x[1], x[2], x[3]))
    limit = cap
    if limit < 1:
        limit = 1
    for _, _, u, v in scored:
        if len(arcs) >= limit:
            break
        arcs.append([int(u), int(v)])
    return subproblems, arcs
