import csv

import numpy as np

csv_path = r"C:\Users\pmih0\OneDrive\Рабочий стол\Моделирование\Лаб1\Вариант\Вариант_207.csv"

with open(csv_path, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)
    data_str = ",".join(row[0].strip() for row in reader if row and row[0].strip())

data = np.array([float(x) for x in data_str.split(',')])

sizes = [10, 20, 50, 100, 200, 300]
t_p = {0.9: 1.643, 0.95: 1.960, 0.99: 2.576}
results = {}

for n in sizes:
    sample = data[:n]
    mean = np.mean(sample)
    var = np.var(sample, ddof=1)
    std = np.sqrt(var)
    
    results[n] = {
        'mean': mean, 
        'var': var, 
        'std': std, 
        'cv': std / mean,
        'eps_90': t_p[0.9] * np.sqrt(var / n),
        'eps_95': t_p[0.95] * np.sqrt(var / n),
        'eps_99': t_p[0.99] * np.sqrt(var / n)
    }

def pct_diff(val, base):
    return abs(val - base) / base * 100

print("ФОРМА 1. Характеристики заданной ЧП")
print("-" * 110)
print(f"{'Характеристика':<20} | {'10':<12} | {'20':<12} | {'50':<12} | {'100':<12} | {'200':<12} | {'300':<12}")
print("-" * 110)

params = [
    ('Мат.ож.', 'mean'), ('Дов. инт. (0,9)', 'eps_90'), ('Дов. инт. (0,95)', 'eps_95'), 
    ('Дов. инт. (0,99)', 'eps_99'), ('Дисперсия', 'var'), ('С.к.о.', 'std'), ('К-т вариации', 'cv')
]

for name, key in params:
    row_vals = f"{name:<12} Знач. | "
    for n in sizes:
        val = results[n][key]
        row_vals += f"±{val:<11.3f} | " if 'eps' in key else f"{val:<12.3f} | "
    print(row_vals)
    
    row_pcts = f"{'':<12} %     | "
    base_val = results[300][key]
    for n in sizes:
        row_pcts += f"{0.00:<12.2f} | " if n == 300 else f"{pct_diff(results[n][key], base_val):<12.2f} | "
    
    print(row_pcts)
    print("-" * 110)