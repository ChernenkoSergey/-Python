# python movie_recommender.py --user "Chris Duncan"
# python movie_recommender.py --user "Julie Hammel"

import argparse
import json
import numpy as np

from compute_scores import pearson_score
from collaborative_filtering import find_similar_users

# Определяем функцию для синтаксического анализа входных аргументов, единственными входными аргументом будет имя пользователя
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find the movie recommendations for the given user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser

# Определяем функцию для получения рекомендаций фильма для данного пользователя
# Если пользователь не существует в наборе данных, вызываем ошибку
def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Cannot find ' + input_user + ' in the dataset')

    # Определяем переменные для отслеживания оценок
    overall_scores = {}
    similarity_scores = {}

    # Вычисляем оценку подобия между пользователем ввода и всеми другими пользователями в наборе данных
    for user in [x for x in dataset if x != input_user]:
        similarity_score = pearson_score(dataset, input_user, user)

        # Если оценка подобия меньше 0, можем продолжить работу со следующим пользователем в наборе данных
        if similarity_score <= 0:
            continue

        # Извлекаем список фильмов, которые были оценены текущим пользователем, но не были оценены пользователем ввода
        filtered_list = [x for x in dataset[user] if x not in \
                dataset[input_user] or dataset[input_user][x] == 0]

        # Для каждого элемента в отфильтрованном списке сохраняем оценку взвешенного рейтинга 
        # на основе оценки подобия, также отслеживаем оценки подобия
        for item in filtered_list:
            overall_scores.update({item: dataset[user][item] * similarity_score})
            similarity_scores.update({item: similarity_score})

    # Если таких фильмов нет, тогда ничего не можем рекомендовать
    if len(overall_scores) == 0:
        return ['No recommendations possible']

    # Нормализуем баллы на основе взвешенных оценок
    movie_scores = np.array([[score/similarity_scores[item], item]
            for item, score in overall_scores.items()])

    # Сортируем оценки и извлекаем рекомендации по фильму
    movie_scores = movie_scores[np.argsort(movie_scores[:, 0])[::-1]]

    movie_recommendations = [movie for _, movie in movie_scores]

    return movie_recommendations

# Определяем основную функцию и анализируем входные аргументы для извлечения имени входного пользователя
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    # Загружаем данные из файла рейтингов фильмов ratings.json
    # Файл содержит имена людей и их рейтинги для различных фильмов
    ratings_file = 'ratings.json'

    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())

    # Извлекаем рекомендации фильма и выводим результат
    print("\nMovie recommendations for " + user + ":")
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i+1) + '. ' + movie)
