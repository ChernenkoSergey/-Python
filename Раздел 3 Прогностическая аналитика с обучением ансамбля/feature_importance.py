import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn import cross_validation
from sklearn.utils import shuffle

# Используем набор встроенных корпусов, доступных в scikit-learn
housing_data = datasets.load_boston()

# Перемешываем данные, чтобы мы не смещали анализ
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# Делим набор данных на обучение и тестирование
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.2, random_state=7)

# Определяем и обучаем AdaBoostRegressor, используя регрессию дерева решений как отдельную модель
regressor = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
        n_estimators=400, random_state=7)
regressor.fit(X_train, y_train)

# Оцениваем производительность регресса
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred )
print("\nADABOOST REGRESSOR")
print("Mean squared error =", round(mse, 2))
print("Explained variance score =", round(evs, 2))

# Регресс имеет встроенный метод, который можно вызвать для вычисления относительной значимости функции
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# Нормализуем значения относительной значимости
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортируем их так, чтобы они могли быть построены
index_sorted = np.flipud(np.argsort(feature_importances))

# Упорядочиваем галочки по оси x для гистограммы
pos = np.arange(index_sorted.shape[0]) + 0.5

# Строим график гистограммы
plt.figure()
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted])
plt.ylabel('Relative Importance')
plt.title('Feature importance using AdaBoost regressor')
plt.show()

