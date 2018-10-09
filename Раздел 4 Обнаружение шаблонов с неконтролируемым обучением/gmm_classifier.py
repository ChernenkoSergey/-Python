import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from sklearn import datasets
from sklearn.mixture import GMM
from sklearn.cross_validation import StratifiedKFold

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# Используем набор диафрагмы, доступный в scikit-learn для анализа
iris = datasets.load_iris()

# Делили набор данных на обучение и тестирование с использованием разделения 80% на 20%
# Параметр n_folds указывает количество подмножеств, которое получим, используем значение 5, 
# что означает, что набор данных будет разделен на пять частей
# Будем использовать четыре части для обучения и одну часть для тестирования, которая дает раскол 80% на 20%
indices = StratifiedKFold(iris.target, n_folds=5)

# Извлекаем данные обучения
train_index, test_index = next(iter(indices))

# Извлекаем данные обучения и метки
X_train = iris.data[train_index]
y_train = iris.target[train_index]

# Извлекаем тестовые данные в метках
X_test = iris.data[test_index]
y_test = iris.target[test_index]

# Извлекаем количество классов в данных обучения
num_classes = len(np.unique(y_train))

# Строим классификатор на основе GMM, используя соответствующие параметры
# Параметр n_components указывает количество компонентов в базовом дистрибутиве. В этом случае это число различных классов в данных
# Нужно указать тип ковариации для использования. В этом случае используем полную ковариацию
# Параметр init_params управляет параметрами, которые необходимо обновить во время учебного процесса, используем wc, что означает, 
# что параметры веса и ковариации будут обновляться во время обучения
# Параметр n_iter относится к числу итераций ожиданий и максимизации, которые будут выполняться во время обучения
classifier = GMM(n_components=num_classes, covariance_type='full',
        init_params='wc', n_iter=20)

# Инициализируем средства классификатора
classifier.means_ = np.array([X_train[y_train == i].mean(axis=0)
                              for i in range(num_classes)])

# Обучаем классификатор модели смеси Гаусса с использованием данных обучения
classifier.fit(X_train)

# Визуализируем границы классификатора, будем извлекать собственные значения и 
# собственные векторы для оценки того, как рисовать эллиптические границы вокруг кластеров
plt.figure()
colors = 'bgr'
for i, color in enumerate(colors):
    # Извлекаем собственные значения и собственные векторы
    eigenvalues, eigenvectors = np.linalg.eigh(
            classifier._get_covars()[i][:2, :2])

    # Код нормализует первый собственный вектор
    norm_vec = eigenvectors[0] / np.linalg.norm(eigenvectors[0])

    # Эллипсы необходимо поворачивать для точного отображения распределения этот код оценивает угол
    angle = np.arctan2(norm_vec[1], norm_vec[0])
    angle = 180 * angle / np.pi

    # Увеличиваем эллипсы для визуализации
    scaling_factor = 8
    eigenvalues *= scaling_factor

    # Собственные значения управляют размером эллипсов, рисуем эллипсы
    ellipse = patches.Ellipse(classifier.means_[i, :2],
            eigenvalues[0], eigenvalues[1], 180 + angle,
            color=color)
    axis_handle = plt.subplot(1, 1, 1)
    ellipse.set_clip_box(axis_handle.bbox)
    ellipse.set_alpha(0.6)
    axis_handle.add_artist(ellipse)

# Накладываем входные данные на график
colors = 'bgr'
for i, color in enumerate(colors):
    cur_data = iris.data[iris.target == i]
    plt.scatter(cur_data[:,0], cur_data[:,1], marker='o',
            facecolors='none', edgecolors='black', s=40,
            label=iris.target_names[i])

    # Накладываем тестовые данные на график
    test_data = X_test[y_test == i]
    plt.scatter(test_data[:,0], test_data[:,1], marker='s',
            facecolors='black', edgecolors='black', s=40,
            label=iris.target_names[i])

# Вычисляем прогнозируемый результат для обучения и тестирования данных
y_train_pred = classifier.predict(X_train)
accuracy_training = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
print('Accuracy on training data =', accuracy_training)

y_test_pred = classifier.predict(X_test)
accuracy_testing = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
print('Accuracy on testing data =', accuracy_testing)

plt.title('GMM classifier')
plt.xticks(())
plt.yticks(())

plt.show()
