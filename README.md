# Ejercicio de análisis y predicción del precio de los vehículos

## Getting Started

Este repositorio contiene un archivo `requirements.txt` que especifica las dependencias necesarias para el proyecto. A continuación se detallan los pasos para crear un entorno virtual de Python y instalar estas dependencias.

1. Clonar el repositorio

Clona este repositorio en tu máquina local utilizando el siguiente comando:

   ```bash
   git clone https://github.com/DavidSolan0/precio_vehiculo.git
   ```

2. Crear un entorno virtual

Navega al directorio del proyecto clonado e inicia un nuevo entorno virtual de Python. Puedes hacerlo con el siguiente comando en la terminal:

   ```bash
   python -m venv venv
   ```

3. Activar el ambiente virtual

   - On Windows:

     ```bash
     venv\Scripts\activate
     ```

   - On macOS and Linux:

     ```bash
     source venv/bin/activate
     ```

4. Instalar las dependencias requeridas desde el archivo `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

## ¿Cómo navegar dentro del proyecto?

### Proyecto de Preprocesamiento de Datos

Este repositorio contiene un conjunto de scripts y herramientas para el preprocesamiento de datos antes de su uso en modelos de aprendizaje automático.

#### Estructura del Proyecto

El proyecto está estructurado de la siguiente manera:

```
precio_vehiculos/
│
├── src/
│ ├── cluster_categorical_features.py
│ └── data_model_preparation.py
│ └── utils.py
│
├── artifacts/
│
├── nb
│
├── README.md
└── requirements.txt
```

- **src/**: Esta carpeta contiene los scripts de Python para el preprocesamiento de datos.
  - `cluster_categorical_features.py`: Contiene una clase para reducir el número de categorías para variables cualitativas con muchas categorías.
  - `data_model_preparation.py`: Contiene un pipeline para preparar los datos para el entrenamiento de modelos.
  
- **artifacts/**: Esta carpeta contendrá los artefactos generados durante el preprocesamiento de datos, como modelos entrenados, transformadores y otros archivos.

- **README.md**: Este archivo README proporciona información sobre cómo configurar el entorno de desarrollo y cómo ejecutar los scripts.

- **requirements.txt**: Archivo que especifica las dependencias necesarias para el proyecto.

- **nb.ipynb**: Notebook que contiene la solución del problema.

#### Uso de los Scripts

##### `cluster_categorical_features.py`

Este script contiene una clase para disminuir el número de categorías para variables cualitativas con muchas categorías, haciendo que sean más adecuadas para el análisis.

Para usar este script, puedes importar la clase `ClusterCategoricalFeatures` en tu propio código Python y crear una instancia de esta clase para aplicar el clustering a tus variables cualitativas.

##### `data_model_preparation.py`

Este script contiene un pipeline para preparar los datos para el entrenamiento de modelos de aprendizaje automático. Incluye pasos como la división de los datos en conjuntos de entrenamiento, validación y prueba, la imputación de valores faltantes, la codificación de características categóricas y el escalado de características numéricas.

Para usar este script, puedes ejecutarlo directamente desde la línea de comandos o importar las funciones relevantes en tu propio código Python.

Si tienes alguna pregunta o problema, no dudes en abrir un issue en este repositorio.



