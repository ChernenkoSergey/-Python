import numpy as np
from sklearn import datasets
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.utils import shuffle

# Загружаем набор данных
data = datasets.load_boston()

# Перетасовываем данные
X, y = shuffle(data.data, data.target, random_state=7)

# Разделяем набор данных на обучение и тестирование в формате 80% / 20%
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]

# Cоздаем и обучаем опорные векторы агрессора с использованием линейного ядра
# Параметр C представляет собой штраф за ошибку обучения
# Параметр epsilon указывает пороговое значение
sv_regressor = SVR(kernel='linear', C=1.0, epsilon=0.1)

# Тренируем опорные векторы агрессора
sv_regressor.fit(X_train, y_train)

# Оцениваем производительность регрессора и распечатываем показатели
y_test_pred = sv_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
evs = explained_variance_score(y_test, y_test_pred)
print("\n#### Performance ####")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Берем тестовую точку данных и выполняем прогнозирование
test_data = [3.7, 0, 18.4, 1, 0.87, 5.95, 91, 2.5052, 26, 666, 20.2, 351.34, 15.27]
print("\nPredicted price:", sv_regressor.predict([test_data])[0])