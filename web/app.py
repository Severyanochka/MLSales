from flask import Flask, request, jsonify, render_template
import numpy as np
import main
# экземпляр класса
app = Flask(__name__)

# Загрузка модели и трансформеров
model = main.load_model()
poly = main.load_poly()
scaler = main.load_scaler()

# декоратор определяет маршрут и обращается к html файлу
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()  # преобразование отправленных пользователем данных в словарь методом фласка

    try:
        features = [
            float(data.get('year', 0)),
            float(data.get('month', 0)),
            float(data.get('day', 0)),
            float(data.get('country', 0)),
            float(data.get('sessionID', 0)),
            float(data.get('page1', 0)),
            float(data.get('page2', 0)),
            float(data.get('colour', 0)),
            float(data.get('location', 0)),
            float(data.get('modelPhotography', 0)),
            float(data.get('price', 0)),
            float(data.get('price2', 0)),
            float(data.get('page', 0))
        ]
        features = np.array(features).reshape(1, -1)  # Преобразование в массив с одной строкой и столбцов по кол. элем.

        features_poly = poly.transform(features)
        features_poly_scaled = scaler.transform(features_poly)

        prediction = model.predict(features_poly_scaled)  # Методы перетаскиваются из файла main, без явного импорта

        return jsonify({'prediction': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
