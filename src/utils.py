# Utils para el ejercicio de estimación
import numpy as np
import pandas as pd


def r_cuadrado_ajustado(model, x_scaled, y):
    """
    Calcular el R-cuadrado ajustado para la regresión líneal.

    Parameters:
        model : modelo de regresión de sklearn
            Modelo ajustado de regresión líneal (scikit-learn).
        x_scaled : array-like
            Matriz de covariables.
        y : array-like
            Variable objetivo.

    Returns:
        adjusted_r_squared : float
            Adjusted R-squared value.
    """
    # Calcular el R-cuadrado ajustado
    r_squared = model.score(x_scaled, y)

    # Número de observaciones
    n = len(y)

    # Número de predictores
    p = x_scaled.shape[1]

    # Calcular el R-cuadrado ajustado
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

    return adjusted_r_squared


def probabilidad_de_compra(y_real: np.array, y_pred: np.array) -> np.array:
    """
    Calcula la probabilidad de compra para cada elemento en una lista de predicciones.

    Params:
        y_real (np.array): Array de numpy que contiene los valores reales.
        y_pred (np.array): Array de numpy que contiene los valores predichos.

    Returns:
        np.array: Un array de numpy que contiene las probabilidades de compra para cada elemento.
    """
    # Calcula el cociente entre las predicciones y los valores reales
    ratios = y_pred / y_real

    # Limita los ratios a un mínimo de 1
    ratios = np.maximum(ratios, 1)

    # Calcula la probabilidad de compra basada en la comparación entre predicciones y valores reales
    prob_list = np.where(ratios == 1, 1, 1 - np.minimum(1, ratios - 1))

    return pd.Series(prob_list)
