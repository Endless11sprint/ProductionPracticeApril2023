import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
df = pd.read_csv('car_data.csv')
# построение гистограммы для определения для определения моделей машин, которые потребляют больше топлива
model_groups = df.groupby('car_model')
for model, group in model_groups:
    plt.hist(group['fuel_consumed'], alpha=0.5, label=model)
plt.legend(loc='upper right')
plt.show()
# используем метод IsolationForest для нахождения выбросов
X = df[['fuel_octane', 'fuel_consumed', 'time_spent']]
clf = IsolationForest(random_state=0).fit(X)
y_pred = clf.predict(X)
df['outlier'] = y_pred
# определение подозрительных поездок (outlier равно -1)
suspicious_trips = df[df['outlier'] == -1]
# построение графика для каждой модели автомобиля, чтобы выделить подозрительные поездки
model_groups = suspicious_trips.groupby('car_model')
for model, group in model_groups:
    plt.scatter(group['fuel_consumed'], group['time_spent'], alpha=0.5, label=model)
plt.legend(loc='upper right')
# в результате был получим график, на котором выделены подозрительные поездки для каждой модели автомобиля
plt.show()