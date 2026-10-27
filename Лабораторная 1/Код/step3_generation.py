import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

csv_path = r"C:\Users\pmih0\OneDrive\Рабочий стол\Моделирование\Лаб1\Вариант\Вариант_207.csv"

with open(csv_path, "r", encoding="utf-8") as file:
    reader = csv.reader(file)
    next(reader)
    data_str = ",".join(row[0].strip() for row in reader if row and row[0].strip())

data = np.array([float(x) for x in data_str.split(',')])

q, t1, t2 = 0.2319, 44.3201, 1.6122

np.random.seed(42)
gen_data = np.array([
    np.random.exponential(t1) if np.random.rand() < q else np.random.exponential(t2) 
    for _ in range(300)
])

sizes = [10, 20, 50, 100, 200, 300]
t_p = {0.9: 1.643, 0.95: 1.960, 0.99: 2.576}

def get_stats(arr):
    res = {}
    for n in sizes:
        sample = arr[:n]
        mean = np.mean(sample)
        var = np.var(sample, ddof=1)
        res[n] = {
            'mean': mean, 'var': var, 
            'std': np.sqrt(var), 'cv': np.sqrt(var) / mean,
            'eps_90': t_p[0.9] * np.sqrt(var / n),
            'eps_95': t_p[0.95] * np.sqrt(var / n),
            'eps_99': t_p[0.99] * np.sqrt(var / n)
        }
    return res

orig_stats = get_stats(data)
gen_stats = get_stats(gen_data)

print("ФОРМА 2. Характеристики сгенерированной случайной ЧП")
print("-" * 110)
print(f"{'Характеристика':<20} | {'10':<12} | {'20':<12} | {'50':<12} | {'100':<12} | {'200':<12} | {'300':<12}")
print("-" * 110)

params = [
    ('Мат.ож.', 'mean'), ('Дов. инт. (0,9)', 'eps_90'), ('Дов. инт. (0,95)', 'eps_95'), 
    ('Дов. инт. (0,99)', 'eps_99'), ('Дисперсия', 'var'), ('С.к.о.', 'std'), ('К-т вариации', 'cv')
]

for name, key in params:
    row_vals = f"{name:<12} Знач. | "
    row_pcts = f"{'':<12} %     | "
    for n in sizes:
        g_val, o_val = gen_stats[n][key], orig_stats[n][key]
        row_vals += f"±{g_val:<11.3f} | " if 'eps' in key else f"{g_val:<12.3f} | "
        row_pcts += f"{abs(g_val - o_val) / o_val * 100:<12.2f} | "
        
    print(row_vals)
    print(row_pcts)
    print("-" * 110)

lags = range(1, 11)
orig_acf = [pd.Series(data).autocorr(lag=k) for k in lags]
gen_acf = [pd.Series(gen_data).autocorr(lag=k) for k in lags]

print(f"\nКоэффициент взаимной корреляции: {np.corrcoef(data, gen_data)[0, 1]:.4f}")

print("\nФОРМА 3. Коэффициенты автокорреляции (сравнение исходной и сгенерированной ЧП)")
print("-" * 100)
print("Сдвиг ЧП          | " + " | ".join([f"{i:<7}" for i in lags]))
print("К-т АК задан.      | " + " | ".join([f"{v:<7.3f}" for v in orig_acf]))
print("К-т АК сгенерир.   | " + " | ".join([f"{v:<7.3f}" for v in gen_acf]))
print("%                  | " + " | ".join(
    [f"{abs(g - o) / abs(o) * 100:<7.2f}" for g, o in zip(gen_acf, orig_acf)]
))
print("-" * 100)
print(
    "Примечание: т.к. коэффициенты автокорреляции исходной ЧП близки к нулю, "
    "относительное отклонение (%) для них крайне велико и статистически малоинформативно "
    "— здесь важен сам факт того, что оба ряда коэффициентов лежат в узком коридоре около нуля."
)

out_dir = Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)

fig4, ax4 = plt.subplots(figsize=(7, 5))
ax4.plot(lags, gen_acf, marker='o', linestyle='-', color='g')
ax4.set_title('Автокорреляционный анализ (сгенерированная ЧП)')
ax4.set_xlabel('Сдвиг')
ax4.set_ylabel('Коэффициент автокорреляции')
ax4.grid(True)
fig4.tight_layout()
fig4.savefig(out_dir / "graph_acf_gen.png", dpi=300)

fig5, ax5 = plt.subplots(figsize=(7, 5))
ax5.hist(gen_data, bins=25, color='lightgreen', edgecolor='black', alpha=0.7, density=True)
ax5.set_title('График 3: Гистограмма распределения частот (сгенерированная ЧП)')
ax5.set_xlabel('Интервалы значений')
ax5.set_ylabel('Частота (Плотность)')
ax5.grid(True)
fig5.tight_layout()
fig5.savefig(out_dir / "graph_3.png", dpi=300)

gen_bin_counts, gen_bin_edges = np.histogram(gen_data, bins=25)
print("\nТаблица интервалов (сгенерированная ЧП)")
print("-" * 60)
print(f"{'№':<4} | {'Левая гр.':<12} | {'Правая гр.':<12} | {'Частота':<8}")
print("-" * 60)
for i, cnt in enumerate(gen_bin_counts):
    print(f"{i+1:<4} | {gen_bin_edges[i]:<12.3f} | {gen_bin_edges[i+1]:<12.3f} | {cnt:<8}")
print("-" * 60)

gen_file_path = out_dir / "generated_sequence.txt"
with open(gen_file_path, "w", encoding="utf-8") as f:
    f.writelines(f"{val:.5f}\n" for val in gen_data)

print(f"\nСгенерированная последовательность успешно сохранена в: {gen_file_path}")