import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier

from utilities import visualize_classifier

# Файл содержащий данные
# В файле каждая строка содержит значения, разделенные запятыми, 
# первые два значения соответствуют входным данным, а последнее значение соответствует целевой метке
input_file = 'data_decision_trees.txt'
# Загружаем данные из файла
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разделяем входные данные на два отдельных класса на основе меток
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Визуализируем входные данные с помощью диаграммы рассеяния
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black',
        edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white',
        edgecolors='black', linewidth=1, marker='o')
plt.title('Input data')

# Разбиваем данные на обучение и тестирование наборов данных
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        X, y, test_size=0.25, random_state=5)

# Создаем, строим и визуализируем классификатор дерева решений на основе набора учебных данных
# Параметр random_state относится к семени, используемому генератором случайных чисел, необходимым для инициализации алгоритма классификации дерева решений
# Параметр max_depth относится к максимальной глубине дерева, которое мы хотим построить
params = {'random_state': 0, 'max_depth': 4}
classifier = DecisionTreeClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset')

# Вычисляем результат классификатора в тестовом наборе данных и визуализируем его
y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset')

# Оцениваем производительность классификатора, распечатываем отчет о классификации
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