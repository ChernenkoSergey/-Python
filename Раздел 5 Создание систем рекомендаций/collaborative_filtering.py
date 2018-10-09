# python collaborative_filtering.py --user "Bill Duffy"
# python collaborative_filtering.py --user "Clarissa Jackson"

import argparse
import json
import numpy as np

from compute_scores import pearson_score

# Определяем функцию для анализа входных аргументов, единственными входными аргументом будет имя пользователя
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser

# Определяем функцию, чтобы найти пользователей в наборе данных, которые похожи на данного пользователя
# Если пользователь не находится в наборе данных, вызываем ошибку
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # Уже импортировали функцию для вычисления оценки Пирсона, используем эту функцию 
    # для вычисления оценки Пирсона между пользователем ввода и всеми другими пользователями в наборе данных
    scores = np.array([[x, pearson_score(dataset, user,
            x)] for x in dataset if x != user])

    # Сортируем оценки в порядке убывания
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Извлекаем верхние num_users, количество пользователей, заданное входным аргументом, и возвращаем массив
    top_users = scores_sorted[:num_users]

    return scores[top_users]

# Определяем основную функцию и анализируем входные аргументы для извлечения имени пользователя
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    # Загружаем данные из файла рейтингов фильмов ratings.json
    # Файл содержит имена людей и их рейтинги для различных фильмов
    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    # Находим тройку пользователей, которые похожи на пользователя, заданного входным аргументом
    # Можем изменить его на любое количество пользователей в зависимости от выбора
    # Печатаем результат вместе с оценками
    print('\nUsers similar to ' + user + ':\n')
    similar_users = find_similar_users(data, user, 3)
    print('User\t\t\tSimilarity score')
    print('-'*41)
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))
