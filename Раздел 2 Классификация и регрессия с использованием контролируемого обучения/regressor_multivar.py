import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures

# Файл содержащий данные, с разделителями-запятыми, чтобы легко загрузить его с помощью вызова одной строки
input_file = 'data_multivar_regr.txt'

# Загружаем данные из файла
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разделяем данные на обучение и тестирование
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train, y_train = X[:num_training], y[:num_training]

X_test, y_test = X[num_training:], y[num_training:]

# Создаем и подготавливаем модель линейной регрессии
linear_regressor = linear_model.LinearRegression()

linear_regressor.fit(X_train, y_train)

# Предсказываем выход для тестового набора данных
y_test_pred = linear_regressor.predict(X_test)

# Распечатываем показатели производительности
print("Linear Regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error =", round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explained variance score =", round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))

# Создать полиномиальный регрессион степени 10, обучаем регрессию в наборе учебных материалов
# Берем примерную точку данных и смотрим, как выполнить прогнозирование
# Первым шагом является преобразование его в многочлен, 
# Точка данных очень близка к точке данных в строке 11 в файле данных, которая составляет 7,66, 6,29, 5,66
# Таким образом, хороший регресс должен прогнозировать результат, близкий к 41.35
polynomial = PolynomialFeatures(degree=10)
X_train_transformed = polynomial.fit_transform(X_train)
datapoint = [[7.75, 6.35, 5.56]]
poly_datapoint = polynomial.fit_transform(datapoint)

# Создаем объект линейной регрессии и выполняем полиномиальное соответствие
# Выполняем прогнозирование с использованием как линейных, так и полиномиальных регрессий, чтобы увидеть разницу
poly_linear_model = linear_model.LinearRegression()
poly_linear_model.fit(X_train_transformed, y_train)
print("\nLinear regression:\n", linear_regressor.predict(datapoint))
print("\nPolynomial regression:\n", poly_linear_model.predict(poly_datapoint))

