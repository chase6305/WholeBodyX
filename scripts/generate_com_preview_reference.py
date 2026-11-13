"""Generate a high-precision unconstrained CoM preview fixture independently.

For a unit jerk in interval i, its position and velocity at node k+1 are
dt**3*(d*d/2+d/2+1/6) and dt**2*(d+1/2), where d=k-i >= 0.
Decimal arithmetic forms the scalar normal equations and Cholesky solution.
No WholeBodyX model, prediction matrix or solver is imported.
"""

import argparse
from decimal import Decimal as D, localcontext
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def generate(horizon=100, digits=70):
    if not 2 <= horizon <= 100 or not 40 <= digits <= 200:
        raise ValueError('horizon must be in [2, 100]; digits must be in [40, 200]')
    n = horizon
    phase = .75 + np.arange(n)*.1
    targets = np.column_stack((.015*np.sin(phase), .01*np.cos(phase)))
    velocities = np.column_stack((.015*np.cos(phase), -.01*np.sin(phase)))
    initial = np.array([[0., 0.], [.01, -.02], [.005, .01]])
    with localcontext() as context:
        context.prec = digits
        dt = D.from_float(.1)
        p = [[D(0)]*n for _ in range(n)]
        v = [[D(0)]*n for _ in range(n)]
        for k in range(n):
            for i in range(k+1):
                d = D(k-i)
                p[k][i] = dt**3*(d*d/2+d/2+D(1)/6)
                v[k][i] = dt**2*(d+D(1)/2)
        h = [[D(0)]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                h[i][j] = h[j][i] = sum(
                    p[k][i]*p[k][j] + D('.01')*v[k][i]*v[k][j]
                    for k in range(max(i, j), n)
                ) + (D('.00001') if i == j else D(0))
        lower = [[D(0)]*n for _ in range(n)]
        for i in range(n):
            for j in range(i+1):
                value = h[i][j] - sum(lower[i][k]*lower[j][k] for k in range(j))
                lower[i][j] = value.sqrt() if i == j else value/lower[j][j]
        solutions, trajectories, residuals = [], [], []
        objective = D(0)
        for axis in range(2):
            q0, v0, a0 = (D.from_float(float(value)) for value in initial[:, axis])
            target = [D.from_float(float(value)) for value in targets[:, axis]]
            velocity = [D.from_float(float(value)) for value in velocities[:, axis]]
            dp = [q0+(k+1)*dt*v0+((k+1)*dt)**2*a0/2-target[k] for k in range(n)]
            dv = [v0+(k+1)*dt*a0-velocity[k] for k in range(n)]
            gradient = [sum(p[k][j]*dp[k]+D('.01')*v[k][j]*dv[k] for k in range(j, n))
                        for j in range(n)]
            y, x = [D(0)]*n, [D(0)]*n
            for i in range(n):
                y[i] = (-gradient[i]-sum(lower[i][j]*y[j] for j in range(i)))/lower[i][i]
            for i in reversed(range(n)):
                x[i] = (y[i]-sum(lower[j][i]*x[j] for j in range(i+1, n)))/lower[i][i]
            residuals.append(max(abs(sum(h[i][j]*x[j] for j in range(n))+gradient[i]) for i in range(n)))
            states = [(q0, v0, a0)]
            q, speed, acceleration = q0, v0, a0
            for k, jerk in enumerate(x):
                q, speed, acceleration = (
                    q+dt*speed+dt**2*acceleration/2+dt**3*jerk/6,
                    speed+dt*acceleration+dt**2*jerk/2,
                    acceleration+dt*jerk,
                )
                states.append((q, speed, acceleration))
                objective += ((q-target[k])**2/2 + D('.005')*(speed-velocity[k])**2
                              + D('.000005')*jerk**2)
            solutions.append([float(value) for value in x])
            trajectories.append([[float(value) for value in row] for row in states])
        return dict(description='Independent Decimal unconstrained jerk-model optimum; verify physical feasibility separately.',
                    decimal_digits=digits, horizon=n, dt=.1, height=.8, gravity=9.81,
                    support_points=[[-.12, -.09], [.12, -.09], [.12, .09], [-.12, .09]],
                    initial=initial.tolist(), targets=targets.tolist(), target_velocities=velocities.tolist(),
                    expected_jerks=np.asarray(solutions).T.tolist(),
                    expected_states=np.asarray(trajectories).transpose(1, 2, 0).tolist(),
                    objective=float(objective), objective_decimal=str(objective),
                    normal_equation_residual_decimal=str(max(residuals)),
                    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def render(fixture):
    """Keep each numeric row together so the fixture remains reviewable."""
    fields = []
    for name, value in fixture.items():
        if isinstance(value, list):
            text = '[\n' + ',\n'.join('    '+json.dumps(row, allow_nan=False) for row in value) + '\n  ]'
        else:
            text = json.dumps(value, allow_nan=False)
        fields.append('  '+json.dumps(name)+': '+text)
    return '{\n'+',\n'.join(fields)+'\n}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--digits', type=int, default=70)
    parser.add_argument('--output', type=Path, default=Path('/tmp/com-preview-free-100.json'))
    args = parser.parse_args()
    began = time.perf_counter()
    fixture = generate(args.horizon, args.digits)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(fixture))
    print(json.dumps(dict(output=str(args.output), horizon=args.horizon, digits=args.digits,
                          wall_s=time.perf_counter()-began, residual=fixture['normal_equation_residual_decimal'])))


if __name__ == '__main__':
    main()
