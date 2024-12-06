"""
Clustering Image Analysis and Prediction Package.

This package provides functionality for clustering image datasets and predicting image clusters 
using KMeans clustering. The package can be used for training a model based on image features 
and predicting the cluster of new images. It also offers utilities for loading, displaying, 
and converting image data.

Modules:
    QDataPipeline: Processes and preprocesses the data from a specified file path or ByteIO object.
    ImageClusteringModel: Handles model training, saving, loading, and prediction for 
    KMeans clustering.
    
Classes:
    QClusterer:
        A class responsible for handling KMeans clustering on image datasets. It 
        includes methods for:
        - Training and saving a KMeans model
        - Loading an existing model
        - Predicting the cluster for a new image
        - Visualizing the clustered images
    
    Methods:
        - preprocess: Processes the raw data to generate features for clustering.
        - train_model: Trains a KMeans model on the image features.
        - predict_label: Predicts the cluster label for a given image or file.
        - display_cluster_images: Displays a grid of images per cluster.
        - convert_to_image: Converts a Matplotlib figure to a PIL Image for further processing.
        - save_model: Saves the trained model to disk.
        - load_model: Loads a previously trained model from disk.

Usage Example:
    # Initialize the pipeline
    qdp = QDataPipeline("/path/to/file.csv")
    qdp.preprocess()

    # Initialize the model, train, and save it
    model = ImageClusteringModel()
    model.train_model(pil_images=qdp.image_data, n_clusters=5)
    model.save_model("/path/to/model.pkl")

    # Load the model and predict labels
    model.load_model("/path/to/model.pkl")
    predicted_label = model.predict_label("/path/to/image.jpg")

    # Display clustered images
    model.display_cluster_images(pil_images=qdp.image_data, labels=model.labels_, n_clusters=5)

Requirements:
    - scikit-learn: Required for KMeans clustering.
    - Pillow: For image manipulation and conversion.
    - Matplotlib: For plotting and visualizing data.
    - joblib: For model persistence (saving and loading).

Raises:
    ValueError: If model prediction is attempted without training.
    IOError: If a file or directory cannot be accessed.

Author:
    Paul MacNichol (paulmacnichol@gmail.com)
"""

import os
import sys
import math
import io
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from keras.applications import VGG16
from keras.models import Model
from keras_preprocessing.image import img_to_array
from PIL import Image
from tqdm import tqdm

# sys.path.insert(0, os.path.join(os.getcwd(), "models"))
# from q_data_pipeline import QDataPipeline

QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from q_data_pipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
try:
    if not QDataPipeline_found:
        from QATCH.QModel.q_data_pipeline import QDataPipeline
    QDataPipeline_found = True
except:
    QDataPipeline_found = False
if not QDataPipeline_found:
    raise ImportError("Cannot find 'QDataPipeline' in any expected location.")


class QClusterer:
    """A class to perform image clustering using a VGG16 model and KMeans.

    Attributes:
        model (Model): The VGG16 model for feature extraction.
        kmeans (KMeans): The KMeans clustering model.
    """

    def __init__(self, model_path: str = None) -> None:
        """Initializes the QClusterer pipeline with a VGG16 model or loads a saved KMeans model.

        Args:
            model_path (str, optional): Path to a saved KMeans model. If provided, the model is
            loaded for prediction.
        """
        # Load pre-trained VGG16 model + higher level layers
        base_model = VGG16(
            weights="imagenet", include_top=False, input_shape=(224, 224, 3)
        )
        self.model = Model(inputs=base_model.input, outputs=base_model.output)

        if model_path:
            print(
                f"[INFO] Operating in prediction mode using KMeans model from {model_path}"
            )
            self.kmeans = joblib.load(model_path)
        else:
            print("[INFO] Operating in training mode.")
            self.kmeans = None

    def convert_to_image(self, figure: plt.figure = None) -> Image.Image:
        """Converts a Matplotlib figure to a PIL Image and returns it.

        Args:
            fig (matplotlib.figure.Figure): The Matplotlib figure to convert.

        Returns:
            PIL.Image.Image: The converted image.
        """

        file_buffer = io.BytesIO()
        figure.savefig(file_buffer)
        file_buffer.seek(0)
        image = Image.open(file_buffer)
        return image

    def load_content(self, data_directory: str = None) -> list:
        """Loads content from a directory.

        Args:
            data_directory (str): Path to the directory containing image files.

        Returns:
            list: List of file paths.
        """
        print(f"[STATUS] Loading content from {data_directory}")
        content = []
        exclude_suffixes = {"_poi.csv", "_lower.csv"}

        for root, _, files in os.walk(data_directory):
            csv_files = [
                file
                for file in files
                if file.endswith(".csv")
                and not any(file.endswith(suffix) for suffix in exclude_suffixes)
            ]
            content.extend(os.path.join(root, file) for file in csv_files)

        return content

    def load_images(self, content: list = None, size: int = -1) -> list:
        """Loads images from a list of file paths.

        Args:
            content (list): List of file paths.
            size (int): Number of images to load.

        Returns:
            list: List of PIL images.
        """
        print("[STATUS] Loading images")
        images = []
        for i, file in enumerate(tqdm(content, desc="<<Loading Images>>")):
            if i >= size:
                print("[INFO] Breaking early")
                break
            qdp = QDataPipeline(file)
            qdp.preprocess()
            data = qdp.__dataframe__["Dissipation"]
            fig, ax = plt.subplots()
            ax.plot(data)
            ax.axis("off")
            image = self.convert_to_image(fig)
            images.append(image)
            plt.close()
        return images

    def preprocess_images(
        self, pil_images: list = None, target_size: tuple = (224, 224)
    ) -> np.ndarray:
        """Preprocesses a list of PIL images.

        Args:
            pil_images (list): List of PIL images.
            target_size (tuple, optional): Target size for resizing. Defaults to (224, 224).

        Returns:
            np.ndarray: Array of processed images.
        """
        print("[STATUS] Preprocessing images")
        processed_images = []
        for img in tqdm(pil_images, desc="<<Preprocessing Images>>"):
            img = img.resize(target_size)
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0
            processed_images.append(img_array)
        return np.vstack(processed_images)

    def extract_features(self, images: np.ndarray = None) -> np.ndarray:
        """Extracts features from images using the VGG16 model.

        Args:
            images (np.ndarray): Array of preprocessed images.

        Returns:
            np.ndarray: Array of extracted features.
        """
        print(f"[INFO] Extracting features using model {self.model}")
        features = self.model.predict(images)
        return features.reshape((features.shape[0], -1))

    def find_optimal_clusters(
        self, features: np.ndarray = None, min_k: int = 2, max_k: int = 10
    ) -> int:
        """Finds the optimal number of clusters using silhouette score.

        Args:
            features (np.ndarray): Array of extracted features.
            min_k (int, optional): Minimum number of clusters to test. Defaults to 2.
            max_k (int, optional): Maximum number of clusters to test. Defaults to 10.

        Returns:
            int: Optimal number of clusters.
        """
        print(f"[INFO] Finding optimal_k with max_k={max_k}")
        silhouette_scores = []
        k_values = range(min_k, max_k + 1)

        for k in tqdm(k_values, desc="<<Optimal K>>"):
            kmeans = KMeans(n_clusters=k)
            predicted_labels = kmeans.fit_predict(features)
            silhouette_scores.append(silhouette_score(features, predicted_labels))

        plt.figure(figsize=(10, 5))
        plt.plot(k_values, silhouette_scores, "bx-")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Analysis For Optimal K")
        plt.show()
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(
            f"[INFO] Optimal number of clusters based on silhouette analysis: {optimal_k}"
        )
        return optimal_k

    def perform_clustering(
        self, features: np.ndarray = None, n_clusters: int = 2
    ) -> np.ndarray:
        """Performs KMeans clustering.

        Args:
            features (np.ndarray): Array of extracted features.
            n_clusters (int): Number of clusters.

        Returns:
            np.ndarray: Cluster labels.
        """
        print(f"[INFO] Clustering with {n_clusters} clusters")
        self.kmeans = KMeans(n_clusters=n_clusters)
        predicted_labels = self.kmeans.fit_predict(features)
        return predicted_labels

    def visualize_clusters(
        self, features: np.ndarray = None, predicted_labels: np.ndarray = None
    ) -> None:
        """Visualizes clusters using a scatter plot.

        Args:
            features (np.ndarray): Array of extracted features.
            labels (np.ndarray): Cluster labels.
        """
        print("[STATUS] Visualizing")
        plt.scatter(features[:, 0], features[:, 1], c=predicted_labels, cmap="viridis")
        plt.title("Clusters of 2D Line Plot Images")
        plt.show()

    def display_cluster_images(
        self,
        pil_images: list = None,
        predicted_labels: np.ndarray = None,
        n_clusters: int = 2,
    ) -> None:
        """Displays a sample of images from each cluster.

        Args:
            pil_images (list): List of PIL images.
            labels (np.ndarray): Cluster labels.
            n_clusters (int): Number of clusters.
        """
        plt.subplots(n_clusters, 1, figsize=(15, n_clusters * 5))

        for i in range(n_clusters):
            cluster_indices = np.where(predicted_labels == i)[0]
            n_images = len(cluster_indices)

            n_cols = math.ceil(math.sqrt(n_images))
            n_rows = math.ceil(n_images / n_cols)

            _, ax = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
            ax = ax.flatten()

            for j, idx in enumerate(cluster_indices):
                sample_image = pil_images[idx]

                if sample_image.mode != "RGB":
                    sample_image = sample_image.convert("RGB")

                ax[j].imshow(sample_image)
                ax[j].axis("off")
                ax[j].set_title(f"Image {j+1}")

            for j in range(n_images, len(ax)):
                ax[j].axis("off")

            plt.suptitle(f"Cluster {i}", fontsize=20)
            plt.show()

    def train(
        self,
        data_dir: str = None,
        min_k: int = 2,
        max_k: int = 10,
        plotting: bool = False,
    ) -> np.ndarray:
        """Trains the clustering model using images from the specified directory.

        Args:
            data_dir (str): Directory containing image files.
            min_k (int, optional): Minimum number of clusters to test. Defaults to 2.
            max_k (int, optional): Maximum number of clusters to test. Defaults to 10.
            plotting (bool, optional): Flag for visualizing clustering
        Returns:
            np.ndarray: Cluster labels for the training images.
        """
        content = self.load_content(data_dir)
        images = self.load_images(content, len(content))

        processed_images = self.preprocess_images(images)

        features = self.extract_features(processed_images)

        optimal_k = self.find_optimal_clusters(features, min_k=min_k, max_k=max_k)

        predicted_labels = self.perform_clustering(features, n_clusters=optimal_k)
        if plotting:
            self.visualize_clusters(features, predicted_labels)

            self.display_cluster_images(images, predicted_labels, n_clusters=optimal_k)

        return predicted_labels

    def predict_label(self, file_buffer: str = None) -> int:
        """Generates an image from the input CSV file, returns it as a PIL image, and predicts
        the cluster label.

        Args:
            csv_path (str): Path to the CSV file containing data for prediction.

        Returns:
            int : An integer value representing the cluster number to which the parameterized CSV.
        """
        if self.kmeans is None:
            raise ValueError(
                "Model is not trained or loaded. Please train the model first."
            )
        file_buffer_2 = file_buffer
        if not isinstance(file_buffer_2, str):
            if hasattr(file_buffer_2, "seekable") and file_buffer_2.seekable():
                file_buffer_2.seek(0)  # reset ByteIO buffer to beginning of stream
            else:
                # ERROR: 'file_buffer_2' must be 'BytesIO' type here, but it's not seekable!
                raise IOError(
                    "Cannot 'seek' stream prior to passing to 'QDataPipeline'."
                )
        else:
            # Assuming 'file_buffer_2' is a string to a file path, this will work fine as-is
            pass
        qdp = QDataPipeline(data_filepath=file_buffer)
        qdp.preprocess()
        data = qdp.__dataframe__["Dissipation"]

        # Generate the image
        fig, ax = plt.subplots()
        ax.plot(data)
        ax.axis("off")
        image = self.convert_to_image(fig)
        plt.close()

        # Preprocess the image for prediction
        img = image.resize((224, 224))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Extract features using the model
        features = self.model.predict(img_array).reshape(1, -1)

        # Predict the cluster label
        predicted_label = self.kmeans.predict(features)[0]

        return predicted_label

    def save_model(self, model_path: str = None) -> None:
        """Saves the trained KMeans model to a file.

        Args:
            model_path (str): Path where the KMeans model should be saved.
        """
        if self.kmeans is None:
            raise ValueError("Model is not trained. Please train the model first.")
        print(f"[STATUS] Saving model as {model_path}")
        joblib.dump(self.kmeans, model_path)


# Example usage
if __name__ == "__main__":
    # Example Usage
    qcr = QClusterer()
    labels = qcr.train("content/training_data/test_clusters", min_k=3, max_k=10)
    qcr.save_model("QModel/SavedModels/cluster.joblib")

    # For prediction
    qcp = QClusterer("QModel/SavedModels/cluster.joblib")
    label = qcp.predict_label(
        "content/dropbox_dump/01154/MM231106W10_Y60P_PROBLEM_D10_3rd.csv"
    )
    print(f"The image belongs to cluster {label}")
