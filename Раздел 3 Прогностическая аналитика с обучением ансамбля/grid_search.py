import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation, grid_search
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report

from utilities import visualize_classifier

# Используем данные, доступные в файле для анализа
input_file = 'data_random_forests.txt'
# Загружаем данные из файла
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Делим данные на три класса
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

# Делим данные на обучающие и тестовые наборы данных
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.25, random_state=5)

# Указываем сетку параметров, которые хотим проверить классификатором
# Обычно сохраняем один параметр постоянным и изменяем другой параметр, делаем наоборот, чтобы определить лучшую комбинацию
# В этом случае хотим найти наилучшие значения для n_estimators и max_depth
parameter_grid = [ {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                   {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
                 ]

# Определяем метрики, которые должен использовать классификатор, чтобы найти наилучшую комбинацию параметров
metrics = ['precision_weighted', 'recall_weighted']

# Для каждой метрики нужно запустить поиск в сетке, где обучаем классификатор для определенной комбинации параметров
for metric in metrics:
    print("\n##### Searching optimal parameters for", metric)

    classifier = grid_search.GridSearchCV(
            ExtraTreesClassifier(random_state=0),
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    # Распечатываем оценку для каждой комбинации параметров
    print("\nGrid scores for the parameter grid:")
    for params, avg_score, _ in classifier.grid_scores_:
        print(params, '-->', round(avg_score, 3))

    print("\nBest parameters:", classifier.best_params_)

    # Распечатываем отчет об эффективности
    y_pred = classifier.predict(X_test)
    print("\nPerformance report:\n")
    print(classification_report(y_test, y_pred))

