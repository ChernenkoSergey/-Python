import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Файл содержит сведения о продажах из ряда магазинов розничной одежды. 
# Цель - определить модели и сегментировать рынок в зависимости от количества единиц, проданных в этих магазинах
# Загружаем данные из файла, так как это CSV-файл, используем считыватель csv для чтения данных и преобразования его в массив NumPy
input_file = 'sales.csv'
file_reader = csv.reader(open(input_file, 'r'), delimiter=',')
X = []
for count, row in enumerate(file_reader):
    if not count:
        names = row[1:]
        continue

    X.append([float(x) for x in row[1:]])

X = np.array(X)

# Оцениваем полосу пропускания входных данных
bandwidth = estimate_bandwidth(X, quantile=0.8, n_samples=len(X))

# Вычисляем кластеризацию со средним сдвигом, обучаем модель среднего сдвига, основанную на предполагаемой ширине полосы
meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_model.fit(X)
# Извлекаем метки и центры каждого кластера
labels = meanshift_model.labels_
cluster_centers = meanshift_model.cluster_centers_
num_clusters = len(np.unique(labels))

# Печатаем количество кластеров и центры кластеров
print("\nNumber of clusters in input data =", num_clusters)

print("\nCenters of clusters:")
print('\t'.join([name[:3] for name in names]))
for cluster_center in cluster_centers:
    print('\t'.join([str(int(x)) for x in cluster_center]))

# Работаем с шестимерными данными. Чтобы визуализировать данные, 
# возьмем двумерные данные, сформированные с использованием второго и третьего измерений
cluster_centers_2d = cluster_centers[:, 1:3]

# Строим центры кластеров
plt.figure()
plt.scatter(cluster_centers_2d[:,0], cluster_centers_2d[:,1],
        s=120, edgecolors='black', facecolors='none')

offset = 0.25
plt.xlim(cluster_centers_2d[:,0].min() - offset * cluster_centers_2d[:,0].ptp(),
        cluster_centers_2d[:,0].max() + offset * cluster_centers_2d[:,0].ptp(),)
plt.ylim(cluster_centers_2d[:,1].min() - offset * cluster_centers_2d[:,1].ptp(),
        cluster_centers_2d[:,1].max() + offset * cluster_centers_2d[:,1].ptp())

plt.title('Centers of 2D clusters')
plt.show()