"""Compare actual WinMark HTML exports with calculate_uir2.py (stdlib only)."""
from pathlib import Path
import html
import importlib.util
import json
import io
from contextlib import redirect_stdout
import re

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('calculation', ROOT.parent / 'app' / 'calculate_uir2.py')
calculation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calculation)
KEYS = {'Y': 'y', 'R': 'rho', 'Q': 'l', 'M': 'm',
        'W': 'w', 'U': 'u', 'Loss': 'pi', 'Perf': 'x'}


def clean(text):
    return html.unescape(re.sub(r'<[^>]*>', '', text)).strip()


def read_export(path):
    source = path.read_bytes().decode('cp1251')
    sections = {clean(h): body for h, body in re.findall(
        r'<H3[^>]*>(.*?)</H3>(.*?)(?=<H3|$)', source, re.I | re.S)}
    tables = []
    for body in re.findall(r'<TABLE\b[^>]*>(.*?)</TABLE>', source, re.I | re.S):
        tables.append([[clean(cell) for cell in re.split(
            r'<T[DH]\b[^>]*>', row, flags=re.I)[1:]]
            for row in re.split(r'<TR\b[^>]*>', body, flags=re.I)[1:]])
    params = {name: float(value) for name, value in re.findall(
        r'(l[123]|mu|nu) = ([0-9.]+)', sections['Элементы'])}
    point, formulas = {}, {}
    for line in sections['Значения характеристик в точке'].split('<br>'):
        line = clean(line)
        if line:
            name, expression, value = line.split(' = ')
            point[name] = float(value)
            formulas[name] = expression
    probabilities = {name: float(value) for name, value in re.findall(
        r'(p\d+) = ([0-9.]+)', sections['Значения стационарных вероятностей в точке'])}
    def numeric_rows(table):
        return {row[0]: list(map(float, row[1:])) for row in table[1:]}
    return dict(params=params, point=point, formulas=formulas,
                probabilities=probabilities, matrix=tables[0],
                varying=numeric_rows(tables[1]),
                metrics=numeric_rows(tables[2]),
                sweep_probabilities=numeric_rows(tables[3]))


def expected(result, name):
    stem = name.rstrip('123')
    suffix = name[len(stem):]
    values = result['nodes'][int(suffix) - 1] if suffix else result['total']
    return values[KEYS[stem]]


def verify(system, export):
    reference = calculation.calculate(system, 0.6)
    states = len(reference['p'])
    metrics_count = (len(reference['nodes']) + 1) * len(KEYS)
    assert len(export['point']) == len(export['formulas']) == metrics_count
    assert len(export['probabilities']) == states
    assert len(export['matrix']) == states
    assert all(len(row) == states for row in export['matrix'])
    assert all(name in export['params'] for name in ('l1', 'l2', 'mu'))
    assert abs(export['params']['mu'] - 0.2) < 1e-12
    if system == 1:
        assert abs(export['params']['nu'] - 0.4) < 1e-12
    weights = calculation.CONFIGS[system][2]
    for i, weight in enumerate(weights, 1):
        assert abs(export['params'][f'l{i}'] - 0.6 * weight) < 1e-12
    matrix = [[export['params'][x] if x in export['params'] else float(x)
               for x in row] for row in export['matrix']]
    errors = dict(matrix=max(abs(x - y) for row, target in zip(matrix, reference['q'])
                             for x, y in zip(row, target)),
                  point_metrics=max(abs(v - expected(reference, name))
                                    for name, v in export['point'].items()),
                  point_probabilities=max(abs(export['probabilities'][f'p{i}'] - p)
                                          for i, p in enumerate(reference['p'])),
                  sweep_metrics=0.0, sweep_probabilities=0.0,
                  formula_values=0.0, probability_normalization=0.0)
    assert len(export['metrics']) == metrics_count
    assert len(export['sweep_probabilities']) == states
    arrays = [*export['metrics'].values(), *export['sweep_probabilities'].values(),
              *export['varying'].values()]
    assert all(len(values) == 40 for values in arrays)
    assert set(export['varying']) == {f'l{i}' for i in range(1, len(weights) + 1)}
    for j in range(40):
        intensity = (j + 1) / 20
        result = calculation.calculate(system, intensity)
        environment = {f'p{i}': p for i, p in enumerate(result['p'])}
        environment.update(export['params'])
        for i, weight in enumerate(weights, 1):
            environment[f'l{i}'] = intensity * weight
            assert abs(export['varying'][f'l{i}'][j] - intensity * weight) < 1e-7
        for name, expression in export['formulas'].items():
            assert re.fullmatch(r'[0-9plmunu+*/().-]+', expression), (name, expression)
            value = eval(expression, {'__builtins__': {}}, environment)
            errors['formula_values'] = max(errors['formula_values'],
                                          abs(value - expected(result, name)))
        for name, values in export['metrics'].items():
            errors['sweep_metrics'] = max(errors['sweep_metrics'],
                                         abs(values[j] - expected(result, name)))
        for i, p in enumerate(result['p']):
            errors['sweep_probabilities'] = max(errors['sweep_probabilities'],
                abs(export['sweep_probabilities'][f'p{i}'][j] - p))
        normalization = abs(sum(v[j] for v in export['sweep_probabilities'].values()) - 1)
        errors['probability_normalization'] = max(errors['probability_normalization'],
                                                 normalization)
    assert errors['matrix'] < 1e-12 and errors['formula_values'] < 1e-10
    assert max(errors[k] for k in ('point_metrics', 'point_probabilities',
               'sweep_metrics', 'sweep_probabilities')) < 1e-5
    return dict(system=system + 1, states=states, metrics=metrics_count, steps=40,
                sweep_metric_values=40 * metrics_count,
                sweep_probability_values=40 * states, errors=errors,
                point_totals={name: export['point'][name] for name in KEYS})


def main():
    results = []
    for system in range(2):
        path = ROOT / f'system{system + 1}' / f'report_system{system + 1}.html'
        result = verify(system, read_export(path))
        results.append(result)
        print(f'SYSTEM_{system + 1}: states={result["states"]}; formulas={result["metrics"]}; steps=40')
        print(f'checked sweep: {result["sweep_metric_values"]} metric values; '
              f'{result["sweep_probability_values"]} probabilities')
        for name, value in result['errors'].items():
            print(f'{name}: max absolute difference={value:.9e}')
        print('point: ' + ' '.join(f'{k}={v:.6f}' for k, v in result['point_totals'].items()))
    print('PASS: both exports agree with independent calculations.')
    (ROOT / 'winmark_verification.json').write_text(
        json.dumps(results, indent=2), encoding='utf8')


if __name__ == '__main__':
    output = io.StringIO()
    try:
        with redirect_stdout(output):
            main()
    finally:
        log = output.getvalue()
        (ROOT / 'winmark_verification.txt').write_text(log, encoding='utf8')
        print(log, end='')
