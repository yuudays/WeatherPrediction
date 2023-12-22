from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import os
class WeatherClassifier:
    def __init__(self):
        self.le = LabelEncoder()
        self.rf_path = "C:/Users/kiril/PycharmProjects/WeatherPrediction/resources/models/rf_model.joblib"
        self.knn_path = "C:/Users/kiril/PycharmProjects/WeatherPrediction/resources/models/knn_model.joblib"
        self.nb_path = "C:/Users/kiril/PycharmProjects/WeatherPrediction/resources/models/nb_model.joblib"

    def preprocess_data(self, df):
        df['weather'] = self.le.fit_transform(df['weather']) # Преобразуем текстовые данные в числовые
        x = df[['temp_min', 'temp_max', 'precipitation', 'wind']] # Признаки для обучения
        y = df['weather'] # Целевая переменная (по сути ответы для обучения)

        # Разделение данных на обучающую и тестовую выборки (80% для обучения, 20 для тестирования)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

        return X_train, X_test, y_train, y_test

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        rf_param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
        }
        rf_model = RandomForestClassifier()
        rf_grid_search = GridSearchCV(rf_model, rf_param_grid, cv=5, refit=True, scoring='accuracy')
        rf_grid_search.fit(X_train, y_train)
        print("Лучшие параметры для модели случайного леса:", rf_grid_search.best_params_)
        rf_best_model = rf_grid_search.best_estimator_
        rf_y_pred = rf_best_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_y_pred)
        print(f"Точность модели случайного леса: {rf_accuracy:.2f}")
        dump(rf_best_model, self.rf_path)
        print(f"Модель случайного леса успешно обучена и сохранена по пути {self.rf_path}")

    def train_knn(self, X_train, y_train, X_test, y_test):
        knn_param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
        }
        knn_model = KNeighborsClassifier()
        knn_grid_search = GridSearchCV(knn_model, knn_param_grid, cv=5, refit=True, scoring='accuracy')
        knn_grid_search.fit(X_train, y_train)
        print("Лучшие параметры для модели ближайших соседей:", knn_grid_search.best_params_)
        knn_best_model = knn_grid_search.best_estimator_
        knn_y_pred = knn_best_model.predict(X_test)
        knn_accuracy = accuracy_score(y_test, knn_y_pred)
        print(f"Точность модели ближайших соседей: {knn_accuracy:.2f}")
        dump(knn_best_model, self.knn_path)
        print(f"Модель ближайших соседей успешно обучена и сохранена по пути {self.knn_path}")

    def train_naive_bayes(self, X_train, y_train, X_test, y_test):
        nb_param_grid = {
            'var_smoothing': [1e-8, 1e-9, 1e-10]
        }
        nb_model = GaussianNB()
        nb_grid_search = GridSearchCV(nb_model, nb_param_grid, cv=5, refit=True, scoring='accuracy')
        nb_grid_search.fit(X_train, y_train)
        print("Лучшие параметры для модели наивного байесовского классификатора:", nb_grid_search.best_params_)
        nb_best_model = nb_grid_search.best_estimator_
        nb_y_pred = nb_best_model.predict(X_test)
        nb_accuracy = accuracy_score(y_test, nb_y_pred)
        print(f"Точность модели наивного байесовского классификатора (Naive Bayes): {nb_accuracy:.2f}")
        dump(nb_best_model, self.nb_path)
        print(f"Модель наивного байесовского классификатора (Naive Bayes) успешно обучена и сохранена по пути {self.nb_path}")



