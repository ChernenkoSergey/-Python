import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Определяем некоторые образцы меток для основной истины и прогнозируемого результата
true_labels = [2, 0, 0, 2, 4, 4, 1, 0, 3, 3, 3]
pred_labels = [2, 1, 0, 2, 4, 3, 1, 0, 1, 3, 3]

# Cоздаем матрицу путаницы, используя только что обозначенные метки
confusion_mat = confusion_matrix(true_labels, pred_labels)

# Визуализируем матрицу путаницы
# Переменная ticks ссылается на количество различных классов
# Есть пять разных ярлыков
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Confusion matrix')
plt.colorbar()
ticks = np.arange(5)
plt.xticks(ticks, ticks)
plt.yticks(ticks, ticks)
plt.ylabel('True labels')
plt.xlabel('Predicted labels')
plt.show()

# Распечатываем отчет о классификации
# В отчете о классификации печатается производительность для каждого класса
targets = ['Class-0', 'Class-1', 'Class-2', 'Class-3', 'Class-4']
print('\n', classification_report(true_labels, pred_labels, target_names=targets))
