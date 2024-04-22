from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
import os
import joblib


class DataPreprocessor:
    """
    Clase para preprocesar datos antes de su uso en modelos de aprendizaje automático.

    Args:
        vehicles (DataFrame): DataFrame que contiene los datos a preprocesar.
        selected_columns (list): Lista de columnas seleccionadas para incluir en el conjunto de
            datos preprocesado.

    Attributes:
        vehicles (DataFrame): DataFrame que contiene los datos a preprocesar.
        selected_columns (list): Lista de columnas seleccionadas para incluir en el conjunto de
            datos preprocesado.
        dataset (DataFrame): Conjunto de datos preprocesado.
        X_train (DataFrame): Conjunto de características de entrenamiento.
        X_val (DataFrame): Conjunto de características de validación.
        X_test (DataFrame): Conjunto de características de prueba.
        y_train (Series): Etiquetas de entrenamiento.
        y_val (Series): Etiquetas de validación.
        y_test (Series): Etiquetas de prueba.
        X_train_encoded (array): Características categóricas codificadas para entrenamiento.
        X_val_encoded (array): Características categóricas codificadas para validación.
        X_test_encoded (array): Características categóricas codificadas para prueba.
        X_train_scaled (array): Características escaladas para entrenamiento.
        X_val_scaled (array): Características escaladas para validación.
        X_test_scaled (array): Características escaladas para prueba.
        imputer_antiguedad (SimpleImputer): Imputador para valores faltantes.
        encoder (OneHotEncoder): Codificador para características categóricas.
        model_scaler (MaxAbsScaler): Escalador para características numéricas.
    """

    def __init__(self, vehicles, selected_columns):
        self.vehicles = vehicles
        self.selected_columns = selected_columns
        self.dataset = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.X_train_encoded = None
        self.X_val_encoded = None
        self.X_test_encoded = None
        self.X_train_scaled = None
        self.X_val_scaled = None
        self.X_test_scaled = None
        self.imputer_antiguedad = None
        self.encoder = None
        self.model_scaler = None

    def split_data(self, test_size=0.3, random_state=None):
        """
        Dividir los datos en conjuntos de entrenamiento, validación y prueba.
        """
        self.dataset = self.vehicles[self.selected_columns]

        y = self.dataset.pop("price")
        X = self.dataset

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state
        )

        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_train, self.y_val, self.y_test = y_train, y_val, y_test

    def impute_missing_values(self):
        """
        Imputar valores faltantes en el conjunto de datos.
        """
        imputer_antiguedad = SimpleImputer(strategy="mean")
        self.X_train["vehicle_age"] = imputer_antiguedad.fit_transform(
            self.X_train[["vehicle_age"]]
        )
        self.X_val["vehicle_age"] = imputer_antiguedad.transform(
            self.X_val[["vehicle_age"]]
        )
        self.X_test["vehicle_age"] = imputer_antiguedad.transform(
            self.X_test[["vehicle_age"]]
        )
        self.imputer_antiguedad = imputer_antiguedad

    def encode_categorical_features(self):
        """
        Codificar características categóricas.
        """
        object_cols = self.X_train.select_dtypes(include=["object"]).columns.tolist()
        encoder = OneHotEncoder(drop="first")

        self.X_train_encoded = encoder.fit_transform(self.X_train[object_cols])
        self.X_val_encoded = encoder.transform(self.X_val[object_cols])
        self.X_test_encoded = encoder.transform(self.X_test[object_cols])

        self.encoder = encoder

    def scale_features(self):
        """
        Escalar características numéricas.
        """
        model_scaler = MaxAbsScaler()

        self.X_train_scaled = model_scaler.fit_transform(self.X_train_encoded)
        self.X_val_scaled = model_scaler.transform(self.X_val_encoded)
        self.X_test_scaled = model_scaler.transform(self.X_test_encoded)

        self.model_scaler = model_scaler

    def save_artifacts(self, output_dir="artifacts"):
        """
        Guardar los artefactos de preprocesamiento.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        joblib.dump(
            self.imputer_antiguedad, os.path.join(output_dir, "imputer_antiguedad.pkl")
        )
        joblib.dump(self.encoder, os.path.join(output_dir, "encoder.pkl"))
        joblib.dump(self.model_scaler, os.path.join(output_dir, "model_scaler.pkl"))

    def get_processed_dataset(self):
        # Partición de los datos
        self.split_data()

        # Imputación de valores faltantes
        self.impute_missing_values()

        # Codificación de características categóricas
        self.encode_categorical_features()

        # Escalamiento de características
        self.scale_features()

        # Guardar los artefactos
        self.save_artifacts()

        return {
            "X_train": self.X_train_scaled,
            "X_val": self.X_val_scaled,
            "X_test": self.X_test_scaled,
            "y_train": self.y_train,
            "y_val": self.y_val,
            "y_test": self.y_test,
        }
