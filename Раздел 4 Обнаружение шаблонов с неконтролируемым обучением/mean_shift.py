import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle

# Загружаем входные данные
X = np.loadtxt('data_clustering.txt', delimiter=',')

# Оцениваем пропускную способность входных данных, полоса пропускания является 
# параметром основного процесса оценки плотности ядра, используемого в алгоритме среднего сдвига
# Параметр квантиля влияет на оценку пропускной способности, более высокое значение для квантиля 
# увеличивает оценочную полосу пропускания, что приводит к меньшему количеству кластеров
bandwidth_X = estimate_bandwidth(X, quantile=0.1, n_samples=len(X))

# Готовим модель кластеризации среднего сдвига с использованием оценочной полосы пропускания
meanshift_model = MeanShift(bandwidth=bandwidth_X, bin_seeding=True)
meanshift_model.fit(X)

# Извлекаем центры всех кластеров
cluster_centers = meanshift_model.cluster_centers_
print('\nCenters of clusters:\n', cluster_centers)

# Оцениваем и извлекаем количество кластеров
labels = meanshift_model.labels_
num_clusters = len(np.unique(labels))
print("\nNumber of clusters in input data =", num_clusters)

# Визуализируем точки данных, для этого строим точки и центры кластеров
plt.figure()
markers = 'o*xvs'
for i, marker in zip(range(num_clusters), markers):
    plt.scatter(X[labels==i, 0], X[labels==i, 1], marker=marker, color='black')

    # Код для построения центра текущего кластера
    cluster_center = cluster_centers[i]
    plt.plot(cluster_center[0], cluster_center[1], marker='o',
            markerfacecolor='black', markeredgecolor='black',
            markersize=15)

plt.title('Clusters')
plt.show()
