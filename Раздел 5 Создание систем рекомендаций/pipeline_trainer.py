from sklearn.datasets import samples_generator
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

# Генерируем некоторые помеченные образцы данных для обучения и тестирования
# Scikit-learn имеет встроенную функцию, которая обрабатывает его. 
# Создаем 150 точек данных, где каждая точка данных является 25-мерным вектором признаков. 
# Цифры в каждом объекте будут генерироваться с использованием генератора случайных выборок. 
# Каждая точка данных имеет 6 информационных функций и не имеет избыточных функций
X, y = samples_generator.make_classification(n_samples=150,
        n_features=25, n_classes=3, n_informative=6,
        n_redundant=0, random_state=7)

# Первый блок в конвейере - это селектор функций, этот блок выбирает лучшие функции K
# Устанавливаем значение K в 9
k_best_selector = SelectKBest(f_regression, k=9)

# Следующий блок в конвейере является чрезвычайно случайным классификатором леса с 60 оценками и максимальной глубиной 4
classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4)

# Строим конвейер путем объединения отдельных блоков, которые создали
# Можем назвать каждый блок так, чтобы его было легче отслеживать
processor_pipeline = Pipeline([('selector', k_best_selector), ('erf', classifier)])

# Можем изменить параметры отдельных блоков, изменяем значение K на 7 и количество оценок до 30
# Используем имена, которые назначили в предыдущей строке, чтобы определить область
processor_pipeline.set_params(selector__k=7, erf__n_estimators=30)

# Обучаем конвейер, используя данные образца, которые сгенерировали ранее
processor_pipeline.fit(X, y)

# Предсказываем результат для всех входных значений и распечатываем его
output = processor_pipeline.predict(X)
print("\nPredicted output:\n", output)

# Вычисляем счет, используя маркированные данные обучения
print("\nScore:", processor_pipeline.score(X, y))

# Хотим извлечь функции, выбранные блоком селектора. Указали, что нам нужно выбрать 7 функций из 25
# Распечатываем функции, выбранные селектором конвейера
status = processor_pipeline.named_steps['selector'].get_support()

# Извлекаем и распечатываем индексы выбранных функций
selected = [i for i, x in enumerate(status) if x]
print("\nIndices of selected features:", ', '.join([str(x) for x in selected]))