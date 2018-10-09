# python compute_scores.py --user1 "David Smith" --user2 "Bill Duffy" --score-type Euclidean
# python compute_scores.py --user1 "David Smith" --user2 "Bill Duffy" --score-type Pearson

import argparse
import json
import numpy as np


# Создаем парсер аргументов для обработки входных аргументов, он принимает двух пользователей 
# и тип оценки, который он должен использовать для вычисления оценки подобия
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=True,
            help='First user')
    parser.add_argument('--user2', dest='user2', required=True,
            help='Second user')
    parser.add_argument("--score-type", dest="score_type", required=True,
            choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser

# Определяем функцию для вычисления евклидовой оценки между входными пользователями
# Если пользователи не находятся в наборе данных, вызовите ошибку
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Определяем переменную для отслеживания фильмов, которые были оценены обоими пользователями
    common_movies = {}

    # Извлекаем фильмы, оцененные обоими пользователями
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # Если нет общих фильмов, то не можем вычислить оценку подобия
    if len(common_movies) == 0:
        return 0

    # Вычисляем квадратные различия между рейтингами и используем их для вычисления евклидовой оценки
    squared_diff = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))

    return 1 / (1 + np.sqrt(np.sum(squared_diff)))

# Определяем функцию для вычисления оценки Пирсона между входными пользователями в данном наборе данных
# Если пользователи не найдены в наборе данных, вызываем ошибку
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')

    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')

    # Определяем переменную для отслеживания фильмов, которые были оценены обоими пользователями
    common_movies = {}

    # Извлекаем фильмы, оцененные обоими пользователями
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1

    # Если нет общих фильмов, то не можем вычислить оценку подобия
    num_ratings = len(common_movies)

    if num_ratings == 0:
        return 0

    # Вычисляем сумму рейтингов всех фильмов, которые были оценены обоими пользователями
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])

    # Вычисляем сумму квадратов рейтингов всех фильмов, которые были оценены пользователями
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])

    # Рассчитываем сумму продуктов рейтингов всех фильмов, оцененных обоими пользователями
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])

    # Вычисляем различные параметры, необходимые для вычисления оценки Пирсона, используя исходящие вычисления
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings

    # Если отклонений нет, то оценка равна нулю
    if Sxx * Syy == 0:
        return 0

    # Возвращаем оценку Пирсона
    return Sxy / np.sqrt(Sxx * Syy)

# Определяем основную функцию и анализируем входные аргументы
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1
    user2 = args.user2
    score_type = args.score_type

    # Загружаем рейтинги из файла ratings.json в словарь
    # Файл содержит имена людей и их рейтинги для различных фильмов
    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    # Вычисляем оценку подобия на основе входных аргументов
    if score_type == 'Euclidean':
        print("\nEuclidean score:")
        print(euclidean_score(data, user1, user2))
    else:
        print("\nPearson score:")
        print(pearson_score(data, user1, user2))
