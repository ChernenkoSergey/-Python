# python random_forests.py --classifier-type rf
# python random_forests.py --classifier-type erf

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import cross_validation
from sklearn.metrics import classification_report

from utilities import visualize_classifier

# Определяем парсер аргументов для Python, чтобы использовать тип классификатора в качестве входного параметра
# В зависимости от этого параметра можно построить случайный классификатор леса или чрезвычайно случайный классификатор леса
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Classify data using \
            Ensemble Learning techniques')
    parser.add_argument('--classifier-type', dest='classifier_type',
            required=True, choices=['rf', 'erf'], help="Type of classifier \
                    to use; can be either 'rf' or 'erf'")
    return parser

# Определяем основную функцию и проанализируем входные аргументы
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

    # Файл содержащий данные
    # В файле каждая строка содержит значения, разделенные запятыми, 
    # первые два значения соответствуют входным данным, а последнее значение соответствует целевой метке
    input_file = 'data_random_forests.txt'
    # Загружаем данные из файла
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Делим входные данные на три класса
    class_0 = np.array(X[y==0])
    class_1 = np.array(X[y==1])
    class_2 = np.array(X[y==2])

    # Визуализируем входные данные
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white',
                    edgecolors='black', linewidth=1, marker='s')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
                    edgecolors='black', linewidth=1, marker='o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white',
                    edgecolors='black', linewidth=1, marker='^')
    plt.title('Input data')

    # Делим данные на учебные и тестовые наборы данных
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(
            X, y, test_size=0.25, random_state=5)

    # Определяем параметры, которые будут использоваться при построении классификатора, 
    # параметр n_estimators относится к числу деревьев, которые будут построены
    # Параметр max_depth относится к максимальному количеству уровней в каждом дереве
    # Параметр random_state относится к начальному значению генератора случайных чисел, 
    # необходимого для инициализации алгоритма случайного лесного классификатора
    # В зависимости от входного параметра строим случайный классификатор леса, либо чрезвычайно случайный классификатор леса
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)

    # Тренируем и визуализируем классификатор
    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Training dataset')

    # Вычисляем результат на основе тестового набора данных и визуализируем его
    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Test dataset')

    # Оцениваем производительность классификатора, распечатав классификационный отчет
    class_names = ['Class-0', 'Class-1', 'Class-2']
    print("\n" + "#"*40)
    print("\nClassifier performance on training dataset\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#"*40 + "\n")

    print("#"*40)
    print("\nClassifier performance on test dataset\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#"*40 + "\n")

    # Определяем массив тестовых точек данных
    test_datapoints = np.array([[5, 5], [3, 6], [6, 4], [7, 2], [4, 4], [5, 2]])

    # Вычисляем уверенность
    # Объект классификатора имеет встроенный метод для вычисления меры доверия, 
    # классифицируем каждую точку и вычисляем значения достоверности
    print("\nConfidence measure:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Class-' + str(np.argmax(probabilities))
        print('\nDatapoint:', datapoint)
        print('Predicted class:', predicted_class)

    # Визуализируйте точки данных теста на основе границ классификатора
    visualize_classifier(classifier, test_datapoints, [0]*len(test_datapoints),
            'Test datapoints')

    plt.show()
