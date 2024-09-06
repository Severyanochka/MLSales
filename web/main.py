import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt

# Путь к файлу
file_path = 'C:/Users/1654352/PycharmProjects/MLSales/e-shopСlothing2008.csv'

# Выбор признаков
features = ['year', 'month', 'day', 'country', 'sessionID',
            'page1', 'page2', 'colour', 'location', 'modelPhotography', 'price', 'price2', 'page']


def preprocess_data():
    """
    Загружает данные из CSV файла, обрабатывает их и возвращает признаки и целевую переменную.
    :return:
    X: Признаки для обучения модели.
    y: Целевая переменная для обучения модели.
    """
    data = pd.read_csv(file_path, sep=';')  # Чтение данных из CSV файла с указанием разделителя ';'
    data.ffill(inplace=True)  # Проверка на пропущенные значения и заполнение их предыдущем известным значением
    data['page2'] = data['page2'].astype(
        'category').cat.codes  # Преобразование категориальных признаков в числовые значения
    X = data[features]
    y = data['order']
    return X, y


def save_model_and_transformers(lasso, poly, scaler):
    """
    Сохраняет модель, полиномиальные признаки и стандартизованные признаки для последующего использования
    :param lasso: Обученная модель Lasso
    :param poly: Объект для генерации полиномиальных признаков
    :param scaler: Объект для стандартизации признаков.
    :return:
    """
    """Сохранение модели, полиномиальных признаков и скейлера"""
    with open('model.pkl', 'wb') as f:
        pickle.dump(lasso, f)
    # open('model.pkl', 'wb') открывает файл с именем model.pkl в режиме записи в бинарном формате ('wb'):
    # 'w'- (write).
    # 'b'- (binary).
    # Если файла model.pkl не существует, он будет создан. Если файл уже существует, он будет перезаписан.
    # with используется для автоматического закрытия файла после завершения работы с ним.
    # pickle.dump(obj, file) сериализует (или "упаковывает") объект lasso и записывает его в файл file
    # Модуль pickle преобразует Python-объекты в байтовый поток, который можно сохранить в файл.
    with open('poly.pkl', 'wb') as f:
        pickle.dump(poly, f)

    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)


def load_model():
    """
    Загружает модель Lasso из файла для чтения.
    :return:
    model (Lasso): Загруженная модель Lasso.
    """
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def load_poly():
    """
    Загрузка полиномиальных признаков для чтения
    :return:
    poly (PolynomialFeatures): Загруженный объект для генерации полиномиальных признаков.
    """
    with open('poly.pkl', 'rb') as f:
        poly = pickle.load(f)
    return poly


def load_scaler():
    """
    Загрузка стандартизованых признаков для чтения
    :return:
    poly (PolynomialFeatures): Загруженный объект для генерации полиномиальных признаков.
    """
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler


def train_and_save_model():
    """
    Обучает модель линейной регрессии Lasso на данных, сохраняет модель и трансформеры,
    а также визуализирует результаты предсказания.
    :return:
    y_test: Целевая переменная для тестовой выборки.
    y_pred: Прогнозы модели для тестовой выборки.
    """
    X, y = preprocess_data()
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
    lasso.fit(X_train, y_train)  # Метод fit вычисляет среднее и стандартное отклонение для каждого признака

    save_model_and_transformers(lasso, poly, scaler)

    # Прогнозирование на тестовой выборке
    y_pred = lasso.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    return y_test, y_pred


def predict(features):
    """
    Делает предсказание на основе переданных признаков, используя сохраненную модель и трансформеры.
    :param features: Признаки для предсказания.
    :return:
    prediction: Прогноз модели на основе переданных признаков.
    """
    model = load_model()
    poly = load_poly()
    scaler = load_scaler()

    features_poly = poly.transform(features)
    # метод transform применяет эти вычисленные значения fit для стандартизации данных
    features_poly_scaled = scaler.transform(features_poly)

    prediction = model.predict(features_poly_scaled)
    return prediction

#Вызов функции обучения и сохранения модели
if __name__ == '__main__':
    train_and_save_model()
