"""Variant 33/73/11. Python 3.10+, standard library only.

Run from project root: python app/calculate_uir2.py
Outputs are always written to the project out/ directory.
State: tuple of local (number_in_node, service_phase) pairs.
K includes the job in service; buffer capacity is K - 1.
"""
from itertools import product
from math import prod
import json
import io
from contextlib import redirect_stdout
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "out"

B = 5.0
MU = 1.0 / B
CONFIGS = [((3, 2), (1, 1), (0.75, 0.25)),
           ((2, 1, 2), (2, 1, 1), (0.75, 0.20, 0.05))]


def solve(a, rhs):
    """Gaussian elimination with partial pivoting."""
    n = len(rhs)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(a)]
    for j in range(n):
        k = max(range(j, n), key=lambda i: abs(aug[i][j]))
        assert abs(aug[k][j]) > 1e-14
        aug[j], aug[k] = aug[k], aug[j]
        for i in range(j + 1, n):
            r = aug[i][j] / aug[j][j]
            for k in range(j + 1, n + 1):
                aug[i][k] -= r * aug[j][k]
            aug[i][j] = 0.0
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (aug[i][-1] - sum(aug[i][j] * x[j]
                               for j in range(i + 1, n))) / aug[i][i]
    return x


def local_states(capacity, phases):
    return [(0, 0)] + [(n, f) for n in range(1, capacity + 1)
                      for f in range(1, phases + 1)]


def generator(capacities, phases, rates):
    spaces = [local_states(k, f) for k, f in zip(capacities, phases)]
    states = list(product(*spaces))
    index = {s: i for i, s in enumerate(states)}
    q = [[0.0] * len(states) for _ in states]
    for i, state in enumerate(states):
        for j, (n, f) in enumerate(state):
            moves = []
            if n < capacities[j]:
                target = (1, 1) if n == 0 else (n + 1, f)
                moves.append((target, rates[j]))
            if n > 0:
                if f < phases[j]:
                    target = (n, f + 1)
                else:
                    target = (0, 0) if n == 1 else (n - 1, 1)
                moves.append((target, phases[j] / B))
            for target, rate in moves:
                changed = list(state)
                changed[j] = target
                q[i][index[tuple(changed)]] += rate
        q[i][i] = -sum(q[i])
    return states, q


def stationary(q):
    n = len(q)
    a = [[q[j][i] for j in range(n)] for i in range(n)]
    a[-1] = [1.0] * n
    return solve(a, [0.0] * (n - 1) + [1.0])


def metrics(states, p, capacities, phases, rates):
    nodes = []
    for j, (cap, k, lam) in enumerate(zip(capacities, phases, rates)):
        busy = length = number = loss = output = accept = 0.0
        wait_mass = stay_mass = 0.0
        for state, prob in zip(states, p):
            n, f = state[j]
            busy += prob * (n > 0)
            length += prob * max(0, n - 1)
            number += prob * n
            loss += prob * (n == cap)
            output += prob * (k / B if n > 0 and f == k else 0.0)
            if n < cap:
                # Conditional waiting time given the state seen on arrival.
                # Current service: remaining phases; waiting jobs: full B.
                tw = 0.0 if n == 0 else (k - f + 1) * B / k + (n - 1) * B
                tu = tw + B
                accept += prob
                wait_mass += prob * tw
                stay_mass += prob * tu
        nodes.append(dict(y=lam * B, rho=busy, l=length, m=number,
                          pi=loss, x=output, w=wait_mass / accept,
                          u=stay_mass / accept, a=accept,
                          wait_mass=wait_mass, stay_mass=stay_mass))
    accepted = sum(lam * d['a'] for lam, d in zip(rates, nodes))
    total = {key: sum(d[key] for d in nodes)
             for key in ('y', 'rho', 'l', 'm', 'x')}
    total['pi'] = sum(lam * d['pi'] for lam, d in zip(rates, nodes)) / sum(rates)
    total['w'] = sum(lam * d['wait_mass'] for lam, d in zip(rates, nodes)) / accepted
    total['u'] = sum(lam * d['stay_mass'] for lam, d in zip(rates, nodes)) / accepted
    total['rho_mean'] = total['rho'] / len(nodes)
    return nodes, total


def analytic_local(cap, phases, lam):
    if phases == 1:
        weights = [(lam * B) ** n for n in range(cap + 1)]
    else:
        assert (cap, phases) == (2, 2)
        t = lam * B / 2.0
        weights = [1.0, t * (1 + t), t,
                   t * t * (1 + t), t * t * (2 + t)]
    return [w / sum(weights) for w in weights]


def calculate(system, intensity):
    caps, phases, weights = CONFIGS[system]
    rates = [intensity * w for w in weights]
    states, q = generator(caps, phases, rates)
    p = stationary(q)
    nodes, total = metrics(states, p, caps, phases, rates)
    row_err = max(abs(sum(row)) for row in q)
    residual = max(abs(sum(p[i] * q[i][j] for i in range(len(p))))
                   for j in range(len(p)))
    local = [analytic_local(c, f, a) for c, f, a in zip(caps, phases, rates)]
    spaces = [local_states(c, f) for c, f in zip(caps, phases)]
    factor_err = max(abs(prob - prod(local[j][spaces[j].index(s)]
                                   for j, s in enumerate(state)))
                     for state, prob in zip(states, p))
    checks = [abs(sum(p) - 1), row_err, residual, factor_err]
    for lam, d in zip(rates, nodes):
        checks += [abs(d['x'] - lam * (1 - d['pi'])),
                   abs(d['rho'] - B * d['x']),
                   abs(d['l'] - d['x'] * d['w']),
                   abs(d['m'] - d['x'] * d['u']),
                   abs(d['u'] - d['w'] - B)]
    checks += [abs(total['l'] - total['x'] * total['w']),
               abs(total['m'] - total['x'] * total['u'])]
    assert min(p) >= -1e-12 and max(checks) < 1e-10
    return dict(states=states, q=q, p=p, nodes=nodes, total=total,
                local=local, errors=dict(rows=row_err, stationary=residual,
                                        factorization=factor_err, all=max(checks)))


def placement_study(intensity=0.6):
    """Move E2 service to exactly one of the three servers."""
    caps, _, weights = CONFIGS[1]
    rates = [intensity * w for w in weights]
    results = []
    for chosen in range(3):
        phases = [1, 1, 1]
        phases[chosen] = 2
        states, q = generator(caps, phases, rates)
        p = stationary(q)
        nodes, total = metrics(states, p, caps, phases, rates)
        checks = [abs(sum(p) - 1)]
        checks += [abs(sum(p[i] * q[i][j] for i in range(len(p))))
                   for j in range(len(p))]
        for lam, d in zip(rates, nodes):
            checks += [abs(d['x'] - lam * d['a']),
                       abs(d['l'] - d['x'] * d['w']),
                       abs(d['m'] - d['x'] * d['u'])]
        assert min(p) > 0 and max(checks) < 1e-10
        results.append(dict(server=chosen + 1, states=len(states),
                            total=total, error=max(checks)))
    return results


def crossing():
    left, right = 1.0, 2.0
    for _ in range(50):
        mid = (left + right) / 2
        delta = calculate(0, mid)['total']['rho'] - calculate(1, mid)['total']['rho']
        if delta > 0:
            left = mid
        else:
            right = mid
    return (left + right) / 2


def print_result(label, result):
    print('\n' + label, 'states =', len(result['states']))
    print('i   state                         probability')
    for i, (s, p) in enumerate(zip(result['states'], result['p'])):
        print(f'{i:2d}  {str(s):28s}  {p:.12f}')
    print('node      y        rho        l          m         pi         X          W          U')
    for name, d in list(enumerate(result['nodes'], 1)) + [('ALL', result['total'])]:
        print(f'{str(name):>4s} ' + ' '.join(f'{d[k]:10.6f}'
              for k in ('y', 'rho', 'l', 'm', 'pi', 'x', 'w', 'u')))
    print('max errors:', ' '.join(f'{k}={v:.3e}' for k, v in result['errors'].items()))


def main():
    print('Python', sys.version.split()[0])
    print('Variant 33/73/11; lambda=0.6; B=5; mu=0.2; nu=0.4')
    baseline = [calculate(i, 0.6) for i in range(2)]
    for i, result in enumerate(baseline, 1):
        print_result(f'SYSTEM_{i}', result)
    grid = [n / 20 for n in range(1, 41)]
    sweep = [(a, calculate(0, a)['total'], calculate(1, a)['total']) for a in grid]
    print('\nlambda       R1         R2        pi1        pi2         U1         U2')
    for a, x, y in sweep:
        print(f'{a:.2f} ' + ' '.join(f'{v:.8f}' for v in
              (x['rho'], y['rho'], x['pi'], y['pi'], x['u'], y['u'])))
    cross = crossing()
    print(f'R1=R2 at lambda={cross:.9f}')
    # Additional controls isolate buffer and service-distribution effects.
    controls = []
    for cap, phases in [(2, 2), (2, 1), (3, 1), (1, 1)]:
        states, q = generator((cap,), (phases,), (0.45,))
        ps = stationary(q)
        _, t = metrics(states, ps, (cap,), (phases,), (0.45,))
        controls.append(dict(K=cap, phases=phases, total=t))
    print('\nCONTROL: first node, arrival=0.45')
    for d in controls:
        print(f"K={d['K']}, phases={d['phases']}: " +
              ' '.join(f'{k}={d["total"][k]:.9f}' for k in ('rho', 'pi', 'x', 'w', 'u')))
    placement = placement_study()
    print('\nE2 PLACEMENT: system 2, lambda=0.6')
    for d in placement:
        print(f"server={d['server']}, states={d['states']}: " +
              ' '.join(f'{k}={d["total"][k]:.9f}'
                       for k in ('rho', 'pi', 'x', 'w', 'u')))
    (OUTPUT_DIR / 'results.json').write_text(json.dumps(dict(baseline=baseline,
        sweep=sweep, crossing=cross, controls=controls, placement=placement),
        indent=2), encoding='utf8')


if __name__ == '__main__':
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            main()
    finally:
        log = output.getvalue()
        (OUTPUT_DIR / 'calculation_log.txt').write_text(log, encoding='utf8')
        print(log, end='')
