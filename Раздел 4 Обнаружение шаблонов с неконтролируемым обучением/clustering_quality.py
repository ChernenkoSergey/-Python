import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Используем данные из файла
X = np.loadtxt('data_quality.txt', delimiter=',')

# Инициализируем переменные, массив значений будет содержать список значений, 
# которые хотим перебрать, и найти оптимальное количество кластеров
scores = []
values = np.arange(2, 10)

# Перебираем все значения и строим kmeans модель во время каждой итерации
for num_clusters in values:
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)
    kmeans.fit(X)
    # Код для оценки силуэта текущей модели кластеризации 
	# с использованием эвклидовой метрики расстояния
    score = metrics.silhouette_score(X, kmeans.labels_,
                metric='euclidean', sample_size=len(X))

    # Печатаем оценку силуэта для текущего значения
    print("\nNumber of clusters =", num_clusters)
    print("Silhouette score =", score)

    scores.append(score)

# Чтобы визуализировать оценки силуэта для различных значений рисуем график
plt.figure()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Silhouette score vs number of clusters')

# Извлекаем лучший результат и соответствующее значение для количества кластеров
num_clusters = np.argmax(scores) + values[0]
print('\nOptimal number of clusters =', num_clusters)

# Визуализируем входные данные
plt.figure()
plt.scatter(X[:,0], X[:,1], color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
