import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

# Импортируем функцию из файла utilities.py
from utilities import visualize_classifier

# Определяем входные данные образца с помощью двумерных векторов и соответствующих меток
X = np.array([[3.1, 7.2], [4, 6.7], [2.9, 8], [5.1, 4.5], [6, 5], [5.6, 5], [3.3, 0.4], [3.9, 0.9], [2.8, 1], [0.5, 3.4], [1, 4], [0.6, 4.9]])
y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])

# Создаем объект классификатора логистической регрессии
classifier = linear_model.LogisticRegression(solver='liblinear', C=100)

# Обучаем классификатор с использованием данных, которые мы определили ранее
classifier.fit(X, y)

# Визуализируем производительность классификатора, посмотрев на границы классов
visualize_classifier(classifier, X, y)

