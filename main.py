import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Загрузка данных из CSV-файлов
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Объединение данных в один набор для LabelEncoder
data = pd.concat([train_data, test_data], axis=0)

# Создание экземпляра класса LabelEncoder
le = LabelEncoder()

# Проведение кодирования для всех нечисловых столбцов
for col in ['age', 'gender', 'genre', 'artist']:
    data[col] = le.fit_transform(data[col])

# Разделение данных на обучающую и тестовую выборки
train_data = data.iloc[:8]
test_data = data.iloc[8:].reset_index(drop=True)

# Группировка данных по возрасту, полу, жанру и исполнителю
grouped = train_data.groupby(['age', 'gender', 'genre', 'artist']).agg({'like_dislike': 'mean'}).reset_index()

# Нахождение максимального значения для каждой группы
max_result = grouped.loc[grouped.groupby(['age', 'gender'])['like_dislike'].idxmax()]

# Преобразование кодированных значений обратно в категориальные значения
max_result['age'] = le.inverse_transform(max_result['age'])
max_result['gender'] = le.inverse_transform(max_result['gender'])
max_result['genre'] = le.inverse_transform(max_result['genre'])
max_result['artist'] = le.inverse_transform(max_result['artist'])

# Вывод результата
print(max_result)
