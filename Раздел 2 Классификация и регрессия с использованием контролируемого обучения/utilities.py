import numpy as np
import matplotlib.pyplot as plt

# Создаем определение функции, взяв объект-классификатор, входные данные и метки в качестве входных параметров
# Определяем минимальные и максимальные значения направлений x и y, которые будут использоваться в нашей сетке
def visualize_classifier(classifier, X, y):
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Определяем размер шага для сетки
    mesh_step_size = 0.01

    # Создаем размер шага с использованием минимального и максимального значений
    x_vals, y_vals = np.meshgrid(np.arange(min_x, max_x, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

    # Запускаем классификатор во всех точках сетки
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Изменяем массив результата
    output = output.reshape(x_vals.shape)

    # Создаем график, выбераем цветовую схему и накладываем все точки
    plt.figure()

    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # Указываем границы графиков с использованием минимального и максимального значений, добавляем метки и выводим график
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()
