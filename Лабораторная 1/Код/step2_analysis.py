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

series = pd.Series(data)
lags = range(1, 11)
acf_values = [series.autocorr(lag=k) for k in lags]

print("ФОРМА 3. Коэффициенты автокорреляции заданной ЧП")
print("-" * 90)
print("Сдвиг ЧП       | " + " | ".join([f"{i:<6}" for i in lags]))
print("К-т АК задан.  | " + " | ".join([f"{v:<6.3f}" for v in acf_values]))
print("-" * 90)

t = np.mean(data)
v = np.std(data, ddof=1) / t
q = (2 / (1 + v**2)) * 0.8 
t1 = t * (1 + np.sqrt(((1 - q) / (2 * q)) * (v**2 - 1)))
t2 = t * (1 - np.sqrt((q / (2 * (1 - q))) * (v**2 - 1)))

out_dir = Path(__file__).parent / "out"
out_dir.mkdir(exist_ok=True)

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(data, linestyle='-', marker='o', markersize=3, alpha=0.7, color='b')
ax1.set_title('График 1: Значения заданной ЧП')
ax1.set_xlabel('Номер измерения (i)')
ax1.set_ylabel('Значение X_i')
ax1.grid(True)
fig1.tight_layout()
fig1.savefig(out_dir / "graph_1.png", dpi=300)

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(lags, acf_values, marker='o', linestyle='-', color='b')
ax2.set_title('Автокорреляционный анализ (исходная ЧП)')
ax2.set_xlabel('Сдвиг')
ax2.set_ylabel('Коэффициент автокорреляции')
ax2.grid(True)
fig2.tight_layout()
fig2.savefig(out_dir / "graph_acf_orig.png", dpi=300)

fig3, ax3 = plt.subplots(figsize=(7, 5))
ax3.hist(data, bins=25, color='orange', edgecolor='black', alpha=0.7, density=True)
ax3.set_title('График 2: Гистограмма распределения частот')
ax3.set_xlabel('Интервалы значений')
ax3.set_ylabel('Частота (Плотность)')
ax3.grid(True)
fig3.tight_layout()
fig3.savefig(out_dir / "graph_2.png", dpi=300)

fig3b, ax3b = plt.subplots(figsize=(7, 5))
ax3b.hist(data, bins=25, color='orange', edgecolor='black', alpha=0.7, density=True)
ax3b.set_yscale('log')
ax3b.set_title('График 2 (лог. шкала): Гистограмма распределения частот')
ax3b.set_xlabel('Интервалы значений')
ax3b.set_ylabel('Частота (Плотность), лог. шкала')
ax3b.grid(True, which='both')
fig3b.tight_layout()
fig3b.savefig(out_dir / "graph_2_log.png", dpi=300)

bin_counts, bin_edges = np.histogram(data, bins=25)
print("\nТаблица интервалов (исходная ЧП)")
print("-" * 60)
print(f"{'№':<4} | {'Левая гр.':<12} | {'Правая гр.':<12} | {'Частота':<8}")
print("-" * 60)
for i, cnt in enumerate(bin_counts):
    print(f"{i+1:<4} | {bin_edges[i]:<12.3f} | {bin_edges[i+1]:<12.3f} | {cnt:<8}")
print("-" * 60)