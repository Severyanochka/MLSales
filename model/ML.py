import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt

# Чтение данных из CSV файла с указанием разделителя ';'
file_path = 'C:/Users/1654352/PycharmProjects/MLSales/e-shopСlothing2008.csv'
data = pd.read_csv(file_path, sep=';')

# Проверка первых строк данных
print(data.head())

# Проверка на пропущенные значения и заполнение их предыдущем известным значением
data.ffill(inplace=True)

# Проверка столбцов
print(data.columns)

# Преобразование категориальных признаков в числовые значения
data['page2'] = data['page2'].astype('category').cat.codes

# Выбор признаков и целевой переменной
features = ['year', 'month', 'day', 'country', 'sessionID',
            'page1', 'page2', 'colour', 'location', 'modelPhotography', 'price', 'price2', 'page']
X = data[features]
y = data['order']

# Полиномиальные признаки
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Стандартизация признаков
scaler = StandardScaler()
X_poly_scaled = scaler.fit_transform(X_poly)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели линейной регрессии Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

# Прогнозирование на тестовой выборке
y_pred = lasso.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Визуализация результатов

plt.scatter(y_test, y_pred)
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()