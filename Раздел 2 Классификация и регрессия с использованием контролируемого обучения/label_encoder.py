import numpy as np
from sklearn import preprocessing

# Определяем некоторые примеры меток
input_labels = ['red', 'black', 'red', 'green', 'black', 'yellow', 'white']

# Создаем объект-кодер метки и обучаем его
encoder = preprocessing.LabelEncoder()
encoder.fit(input_labels)

# Распечатываем сопоставление между словами и цифрами
print("\nLabel mapping:")
for i, item in enumerate(encoder.classes_):
    print(item, '-->', i)

# Кодируем набор случайно упорядоченных меток
test_labels = ['green', 'red', 'black']
encoded_values = encoder.transform(test_labels)
print("\nLabels =", test_labels)
print("Encoded values =", list(encoded_values))

# Декодируем случайный набор чисел
encoded_values = [3, 0, 4, 1]
decoded_list = encoder.inverse_transform(encoded_values)
print("\nEncoded values =", encoded_values)
print("Decoded labels =", list(decoded_list))
