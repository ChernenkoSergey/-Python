import pickle

import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Файл содержащий данные, с разделителями-запятыми, чтобы легко загрузить его с помощью вызова одной строки
input_file = 'data_singlevar_regr.txt'

# Загружаем данные из файла
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разделяем данные на обучение и тестирование
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:]

# Создаем объект линейного регрессора и обучаем его с использованием данных обучения
regressor = linear_model.LinearRegression()

regressor.fit(X_train, y_train)

# Прогнозируем вывод тестового набора данных с использованием модели обучения
y_test_pred = regressor.predict(X_test)

# Выводим результат
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Вычисляем показатели производительности для регрессора, сравнив основную истину, 
# которая относится к фактическим результатам, с прогнозируемыми результатами
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# После создания модели, сохраняем ее в файл, для дальнейшего использования
output_model_file = 'model.pkl'

# Сохраняем модель
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Загружаем модель из файла
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Выполняем прогнозирование
y_test_pred_new = regressor_model.predict(X_test)
print("\nNew mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))