import DB_CONNECT as db
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
from querylist import queryCreate,queryDrop,queryInsert

plt.style.use('Solarize_Light2')  # стиль графиков

# streamlit run C:\Users\kiril\PycharmProjects\WeatherPrediction\main.py

# Глобальные переменные
le = LabelEncoder()
nb_model = None
resources_path = "C:/Users/kiril/PycharmProjects/WeatherPrediction/resources"

def load_and_explore_data(df):
    st.subheader("Датасет:")
    num_rows_to_display = st.slider("Число отображаемых строчек", 1, len(df), 4)
    st.write(df.head(num_rows_to_display))

    if st.button("Добавить датасет в БД"):
        db.execute_query(queryCreate)
        data_to_insert = [tuple(row) for row in df.values]
        db.insert_data_into_db(data_to_insert)
    if st.button("Удалить датасет"):
        db.execute_query(queryDrop)
    st.subheader("Описательная статистика данных:")
    st.write(df.describe())

    st.subheader("Преобразуем датасет для построения последующих графиков:")
    transform_dataset(df)
    st.write(df.head(num_rows_to_display))

def transform_dataset(df):
    # Преобразуем столбец 'date' в формат datetime
    df['date'] = pd.to_datetime(df['date'])

    # Извлекаем год и месяц из столбца 'date'
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
def plot_histogram(df, feature):
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.hist(df[feature], bins=20, color="lightblue")

    ax.set_xlabel("Максимальная температура")
    ax.set_ylabel("Количество")

    st.pyplot(fig)

    plt.clf()

def plot_graphics(df, feature, ylabel):
    # Создаем сетку графиков
    g = sns.FacetGrid(df, col='year', col_wrap=2, height=4)

    # На каждом графике строим линейный график для 'month' и выбранного признака (feature)
    g.map(sns.lineplot, 'month', feature)

    g.set_axis_labels('Месяц', ylabel)
    g.set_titles(col_template="{col_name}")

    fig = g.fig
    st.pyplot(fig)

    plt.clf()

def plot_weather_distribution_pie(df, feature, colors):
    # Вычисляем количество вхождений каждой уникальной категории в признаке
    x = df[feature].value_counts()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Построение круговой диаграммы
    ax.pie(x, labels=None, autopct='%1.1f%%', startangle=90, colors=colors)

    # Установка легенды с русскоязычными названиями
    legend_labels = {"rain": "Дождь", "fog": "Туман", "sun": "Солнце", "snow": "Снег", "drizzle": "Мелкий дождь"}
    ax.legend(labels=[legend_labels[label] for label in x.index],
              title="Названия секторов", bbox_to_anchor=(1, 1), frameon = True)

    st.pyplot(fig)
    plt.clf()

def train_and_evaluate_model(df):
    global le, nb_model  # Объявляем глобальные переменные

    df['weather'] = le.fit_transform(df['weather'])

    x = df[['temp_min', 'temp_max', 'precipitation', 'wind']]
    y = df['weather']
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)

    y_pred = nb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    st.subheader("Матрица ошибок:")
    st.write(conf_matrix)
    st.subheader(f"Точность: {accuracy:.2f}")

def predict_weather(temp_min, temp_max, precipitation, wind):
    global df, nb_model

    input_data = pd.DataFrame({
        'temp_min': [temp_min],
        'temp_max': [temp_max],
        'precipitation': [precipitation],
        'wind': [wind]
    })

    predicted_label = nb_model.predict(input_data[['temp_min', 'temp_max', 'precipitation', 'wind']])

    predicted_weather = le.inverse_transform([predicted_label])[0]

    # Русификация прогнозируемой погоды
    weather_mapping = {
        'rain': 'Дождь',
        'fog': 'Туман',
        'sun': 'Солнце',
        'snow': 'Снег',
        'drizzle': 'Мелкий дождь'
    }

    predicted_weather_ru = weather_mapping[predicted_weather]

    return predicted_weather_ru,predicted_weather

def predict():
    st.title(":sunny:Прогноз погоды:umbrella:")

    # Форма для ввода данных
    temp_min = st.slider("Минимальная температура", min_value=-10.0, max_value=40.0, value=15.0,step=0.1)
    temp_max = st.slider("Максимальная температура", min_value=-10.0, max_value=40.0, value=25.0,step=0.1)
    precipitation = st.slider("Осадки", min_value=0.0, max_value=50.0, value=10.0,step=0.1)
    wind = st.slider("Скорость ветра", min_value=0.0, max_value=30.0, value=5.0,step=0.1)

    # Кнопка для выполнения прогнозирования
    checkbox_value = st.checkbox("Добавить данные в бд?")

    if st.button("Прогнозировать погоду"):
        predicted_weather_ru,predicted_weather = predict_weather(temp_min, temp_max, precipitation, wind)
        st.success(f"Прогноз погоды: {predicted_weather_ru}")
        if checkbox_value:
            db.execute_query(queryInsert,precipitation, temp_max, temp_min, wind, predicted_weather)

def gismeteo(image_paths):
    # Разделение на две колонки
    columns = st.columns(2)

    # Вывод изображений в первой колонке
    with columns[0]:
        for i in range(0, 2):
            image = Image.open(image_paths[i])
            if i == 0:
                st.image(image, caption="Погода на сайте", use_column_width=True)
            elif i == 1:
                st.image(image, caption="Прогноз", use_column_width=True)

    # Вывод изображений во второй колонке
    with columns[1]:
        for i in range(2, 4):
            image = Image.open(image_paths[i])
            if i == 2:
                st.image(image, caption="Погода на сайте", use_column_width=True)
            elif i == 3:
                st.image(image, caption="Прогноз", use_column_width=True)

if __name__ == '__main__':
    df = pd.read_csv(resources_path + "/seattle-weather.csv")

    st.title(":rainbow: :rainbow[Прогноз погоды]")

    # Текст с перечислением студентов
    st.markdown("Выполнили студенты группы 1391:")
    st.markdown("- **_Мец Кирилл_**")
    st.markdown("- **_Гречишников Алексей_**")
    st.markdown("- **_Ларьков Никита_**")

    st.header("Анализ датасета")
    load_and_explore_data(df)

    st.subheader("Гистограмма минимальной и максимальной температуры    :")
    columns = st.columns(2)
    with columns[0]:
        plot_histogram(df, 'temp_max')
    with columns[1]:
        plot_histogram(df, 'temp_min')

    st.subheader("График минимальной и максимальной температуры в каждом месяце по годам:")
    columns = st.columns(2)
    with columns[0]:
        plot_graphics(df, 'temp_max', 'Максимальная температура (°C)')
    with columns[1]:
        plot_graphics(df, 'temp_min', 'Минимальная температура (°C)')

    st.subheader("Осадки и скорость ветра в каждом месяце по годам:")
    columns = st.columns(2)
    with columns[0]:
        plot_graphics(df, 'precipitation', 'Осадки')
    with columns[1]:
        plot_graphics(df, 'wind', 'Скорость ветра')

    st.subheader("Распределение типов погоды:")
    plot_weather_distribution_pie(df, 'weather', ['#3498db', '#f39c12', '#95a5a6', '#e74c3c', '#2ecc71'])

    st.title("Обучение модели")
    train_and_evaluate_model(df)

    st.title("Проверка модели")
    st.subheader("Возьмем данные о погоде с сайта gismeteo")
    gismeteo(image_paths = [
        resources_path + "/gismeteo1.png",
        resources_path + "/gismeteo_result1.png",
        resources_path + "/gismeteo2.png",
        resources_path + "/gismeteo_result2.png"
    ])

    st.subheader("Теперь спрогнозируем другую погоду")
    gismeteo(image_paths = [
        resources_path + "/miami1.png",
        resources_path + "/miami_result1.png",
        resources_path + "/miami2.png",
        resources_path + "/miami_result2.png"
    ])

    predict()
