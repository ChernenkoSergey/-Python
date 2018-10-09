import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics

# Загружаем входные данные из файла
X = np.loadtxt('data_clustering.txt', delimiter=',')
# Определяем количество кластеров, прежде чем применять алгоритм kmeans, определяем его как пять
num_clusters = 5

# Визуализируем входные данные, чтобы увидеть, как выглядит спред
plt.figure()
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none',
        edgecolors='black', s=80)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Input data')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())

# Создаем объект kmean с использованием параметров инициализации
# Параметр init представляет собой метод инициализации для выбора начальных центров кластеров
# Вместо того, чтобы выбирать их случайным образом, используем k-means++ для более эффективного выбора этих центров, 
# это гарантирует, что алгоритм сходится быстро
# Параметр n_clusters относится к числу кластеров
# Параметр n_init относится к числу раз, когда алгоритм должен работать до принятия решения о наилучшем результате
kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10)

# Обучаем модель kmeans с входными данными
kmeans.fit(X)

# Чтобы визуализировать границы, создаем сетку точек и оцениваем модель во всех этих точках
# Определяем step_size этой сетки
step_size = 0.01

# Определяем сетку точек и обеспечиваем покрытие всеч значений в входных данных
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_vals, y_vals = np.meshgrid(np.arange(x_min, x_max, step_size),
        np.arange(y_min, y_max, step_size))

# Прогнозируем результаты для всех точек сетки, используя обучаемую модель kmeans
output = kmeans.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

# Код для построения всех значений результатов и цвета каждого региона
output = output.reshape(x_vals.shape)
plt.figure()
plt.clf()
plt.imshow(output, interpolation='nearest',
           extent=(x_vals.min(), x_vals.max(),
               y_vals.min(), y_vals.max()),
           cmap=plt.cm.Paired,
           aspect='auto',
           origin='lower')

# Накладываем входные точки данных поверх этих цветных областей
plt.scatter(X[:,0], X[:,1], marker='o', facecolors='none',
        edgecolors='black', s=80)

# Код для построения центров кластеров, полученных с использованием алгоритма k-средних
cluster_centers = kmeans.cluster_centers_
plt.scatter(cluster_centers[:,0], cluster_centers[:,1],
        marker='o', s=210, linewidths=4, color='black',
        zorder=12, facecolors='black')

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
plt.title('Boundaries of clusters')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()