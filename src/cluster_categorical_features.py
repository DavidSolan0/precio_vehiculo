from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer


class ClusteringAnalyzer:
    """
    Clase para analizar y realizar clustering en un conjunto de datos.

    Args:
        data (DataFrame): El DataFrame que contiene los datos a analizar.
        group_columns (list): Lista de nombres de columnas para agrupar los datos.

    Attributes:
        data (DataFrame): El DataFrame que contiene los datos a analizar.
        group_columns (list): Lista de nombres de columnas para agrupar los datos.
        summary_stats (DataFrame): Estadísticas resumidas por grupo.
        elbow_value (int): El número óptimo de clusters determinado por el método de Elbow.
        kmeans_model (KMeans): Modelo de clustering KMeans.
        scaled_features (array): Características escaladas.
        elbow_visualizer (KElbowVisualizer): Visualizador de Elbow.
        silhouette_visualizer (SilhouetteVisualizer): Visualizador de Silhouette.
    """

    def __init__(self, data, group_columns):
        self.data = data
        self.group_columns = group_columns
        self.summary_stats = None
        self.elbow_value = None
        self.kmeans_model = None
        self.scaled_features = None
        self.elbow_visualizer = None
        self.silhouette_visualizer = None

    def preprocessing(self):
        """
        Calcular el precio medio, mediano y desviación estándar por combinación de columnas.
        """
        # Calcular el precio medio, mediano y desviación estándar por combinación de columnas
        self.summary_stats = (
            self.data.groupby(self.group_columns)["price"]
            .agg(["mean", "median"])
            .reset_index()
        )

    def train(self):
        """
        Entrenar el modelo de clustering KMeans.
        """
        # Escalar los datos para que tengan la misma magnitud
        scaler = StandardScaler()
        self.scaled_features = scaler.fit_transform(
            self.summary_stats[["mean", "median"]]
        )

        # Visualizar el codo (Elbow Method) para determinar el número óptimo de clusters
        self.elbow_visualizer = KElbowVisualizer(KMeans(), k=(2, 11))
        self.elbow_visualizer.fit(self.scaled_features)
        self.elbow_value = self.elbow_visualizer.elbow_value_
        self.elbow_visualizer.show()

        # Aplicar k-means clustering con el número óptimo de clusters
        self.kmeans_model = KMeans(n_clusters=self.elbow_value)
        self.summary_stats["cluster"] = self.kmeans_model.fit_predict(
            self.scaled_features
        )

    def visualize(self):
        """
        Visualizar la métrica de silueta (Silhouette Score).
        """
        # Visualizar la métrica de silueta (Silhouette Score)
        self.silhouette_visualizer = SilhouetteVisualizer(
            self.kmeans_model, colors="yellowbrick"
        )
        self.silhouette_visualizer.fit(self.scaled_features)
        self.silhouette_visualizer.show()

    def merge_clusters(self):
        """
        Fusionar los resultados del clustering con los datos originales.
        """
        # Almacenar el número inicial de filas
        num_filas_inicial = len(self.data)

        # Agregar el cluster a los datos
        self.data = self.data.merge(
            self.summary_stats[self.group_columns + ["cluster"]],
            on=self.group_columns,
            how="left",
        )

        # Renombrar la columna cluster
        self.data.rename(
            columns={"cluster": f"cluster_{'_'.join(self.group_columns)}"}, inplace=True
        )

        # Verificar que el número de filas no se haya alterado
        num_filas_final = len(self.data)
        if num_filas_final != num_filas_inicial:
            raise ValueError(
                "El número de filas de data se ha alterado durante la fusión de clusters."
            )

    def analyze_and_cluster(self):
        """
        Realizar el análisis y clustering de los datos.

        Returns:
            DataFrame: El DataFrame con los resultados del clustering.
        """
        self.preprocessing()
        self.train()
        self.visualize()
        self.merge_clusters()

        # Definir el nombre de la columna de los clusters
        clusters_column_name = f"cluster_{'_'.join(self.group_columns)}"

        # Obtener el tamaño de cada cluster
        print("Tamaño del:", self.summary_stats["cluster"].value_counts())
        print("")
        # Obtener el precio medio de cada cluster
        print(
            "Precio medio dentro del:",
            self.summary_stats.groupby(["cluster"])["mean"].mean(),
        )
        print("")
        # Obtener el precio medio por cada cluster
        print(
            "Precio medio por:",
            self.data.groupby(clusters_column_name).price.mean(),
        )
        print("")
        # Obtener el precio min por cada cluster
        print(
            "Precio min por:",
            self.data.groupby(clusters_column_name).price.min(),
        )
        print("")
        # Obtener el precio max por cada cluster
        print(
            "Precio max por:",
            self.data.groupby(clusters_column_name).price.max(),
        )
        # Definir el cluster como un atributo tipo objeto
        self.data[clusters_column_name] = self.data[clusters_column_name].astype(
            "object"
        )
        return self.data
