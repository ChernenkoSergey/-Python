import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation

from utilities import visualize_classifier

# Файл содержащий данные
input_file = 'data_multivar_nb.txt'

# Загружаем данные из файла
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Создаем экземпляр классификатора Naive Bayes
classifier = GaussianNB()

# Обучаем классификатор с использованием данных обучения
classifier.fit(X, y)

# Запускаем классификатор данных обучения и предсказываем результат
y_pred = classifier.predict(X)

# Вычисляем точность классификатора, сравнив предсказанные значения с истинными метками
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print("Accuracy of Naive Bayes classifier =", round(accuracy, 2), "%")

# Визуализируем производительность
visualize_classifier(classifier, X, y)

# Разделяем данные на подмножества обучения и тестирования, как указано параметром test_size, 
# распределяем 80% для обучения, а оставшиеся 20% - для тестирования
# Подготовливаем классификатор Naive Bayes по этим данным
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2, random_state=3)
classifier_new = GaussianNB()
classifier_new.fit(X_train, y_train)
y_test_pred = classifier_new.predict(X_test)

# Вычисляем точность классификатора и визуализируем производительность
accuracy = 100.0 * (y_test == y_test_pred).sum() / X_test.shape[0]
print("Accuracy of the new classifier =", round(accuracy, 2), "%")

visualize_classifier(classifier_new, X_test, y_test)

# Используем встроенные функции для accuracy_values, precision_values и recall_values, 
# основанных на трехкратной перекрестной проверке
num_folds = 3
accuracy_values = cross_validation.cross_val_score(classifier,
        X, y, scoring='accuracy', cv=num_folds)
print("Accuracy: " + str(round(100*accuracy_values.mean(), 2)) + "%")

precision_values = cross_validation.cross_val_score(classifier,
        X, y, scoring='precision_weighted', cv=num_folds)
print("Precision: " + str(round(100*precision_values.mean(), 2)) + "%")

recall_values = cross_validation.cross_val_score(classifier,
        X, y, scoring='recall_weighted', cv=num_folds)
print("Recall: " + str(round(100*recall_values.mean(), 2)) + "%")

f1_values = cross_validation.cross_val_score(classifier,
        X, y, scoring='f1_weighted', cv=num_folds)
print("F1: " + str(round(100*f1_values.mean(), 2)) + "%")
