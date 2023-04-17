import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("car_data.csv")

# Количество уникальных марок автомобилей
n_car_models = len(df.car_model.unique())

# Количество уникальных путей
n_paths = 0

# Для каждой марки автомобиля
for car_model in df.car_model.unique():
    # Получение данных только для текущей марки автомобиля
    car_data = df[df.car_model == car_model][['fuel_consumed', 'time_spent']]

    # Проведение кластеризации методом K-means
    kmeans = KMeans(n_clusters=10, random_state=0).fit(car_data)

    # Получение меток кластеров для тестовых данных
    test_labels = kmeans.predict(car_data)

    # Определение уникальных путей для машины
    unique_paths = len(set(test_labels))

    # Добавление количества уникальных путей для текущей марки автомобиля к общему числу
    n_paths += unique_paths

# Вывод количества уникальных путей
print(f"Количество уникальных путей: {n_paths}")