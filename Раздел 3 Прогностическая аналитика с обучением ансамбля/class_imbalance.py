# python class_imbalance.py
# python -W ignore class_imbalance.py
# python class_imbalance.py balance

import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report

from utilities import visualize_classifier

# Файл содержащий данные
# В файле каждая строка содержит значения, разделенные запятыми, 
# первые два значения соответствуют входным данным, а последнее значение соответствует целевой метке
input_file = 'data_imbalance.txt'
# Загружаем данные из файла
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Делим входные данные на два класса
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Визуализируем входные данные с использованием графика рассеяния
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
                edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

# Делим данные на учебные и тестовые наборы данных
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.25, random_state=5)

# Определяем параметры для чрезвычайно случайного классификатора леса
# Обратите внимание, что есть входной параметр, называемый balance, который контролирует, хотим ли мы алгоритмически учитывать дисбаланс классов
# Если это так, нужно добавить еще один параметр class_weight, который сообщает классификатору, что он должен сбалансировать вес так, 
# чтобы он был пропорционален количеству точек данных в каждом классе
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'class_weight': 'balanced'}
    else:
        raise TypeError("Invalid input argument; should be 'balance'")

# Создаем, тренируем и визуализируем классификатор с использованием данных обучения
classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

# Предсказываем результат для тестового набора данных и визуализируем результат
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

# Вычисляем производительность классификатора и печатаем отчет о классификации
class_names = ['Class-0', 'Class-1']
print("\n" + "#"*40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()
