import pandas as pd
import sqlite3
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Подключение к базе данных
try:
    conn = sqlite3.connect('real_estate.db')
    data = pd.read_sql_query("SELECT city, rooms, floor, area, price FROM apartments WHERE status = 'active'", conn)
    conn.close()
    print("Столбцы в данных:", data.columns.tolist())
    print("Форма данных:", data.shape)
except Exception as e:
    print(f"Ошибка загрузки данных: {e}")
    exit()

# Проверка на пустые данные
if data.empty:
    print("Ошибка: Таблица 'apartments' пуста или не содержит записей со статусом 'active'.")
    exit()

# Обработка пропущенных значений
data = data.dropna()
if data.empty:
    print("Ошибка: Нет валидных данных после очистки.")
    exit()

# Обработка выбросов
data = data[data['price'] > 0]
data = data[data['area'] > 10]
data = data[data['rooms'].isin([1, 2, 3, 4])]
data = data[data['floor'] > 0]

# Подготовка данных
X = data[['city', 'rooms', 'floor', 'area']]
y = data['price']

# Кодирование категориальных данных
le = LabelEncoder()
X['city_encoded'] = le.fit_transform(X['city'])

# Создание модели
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X[['city_encoded', 'rooms', 'floor', 'area']], y)

# Сохранение модели и кодировщика
with open('model.pkl', 'wb') as file:
    pickle.dump({'model': model, 'label_encoder': le}, file)

print("Модель обучена и сохранена в model.pkl")