from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve


class AbstractModel:
    def __init__(self):
        self.model = None

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate_f1(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        return f1_score(y_test, y_pred, average="macro")

    def evaluate_f1_t(self, X_test, y_test, threshold=0.5):
        y_proba = self.model.predict_proba(X_test)
        predictions = (y_proba[:, 1] >= threshold).astype(int)
        mask = y_proba.max(axis=1) >= threshold
        y_test_array = np.array(y_test)  # Convertir y_test a un array de NumPy
        y_test_filt = y_test_array[mask]
        y_pred_filt = predictions[mask]
        return (
            f1_score(y_test_filt, y_pred_filt, average="macro"),
            y_test_filt,
            y_pred_filt,
        )

    def roc_curve(self, X_test, y_test):
        y_prob = self.model.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
        return fpr, tpr, thresholds

    def predict(self, X_test):
        return self.model.predict(X_test)


class Linear(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = LogisticRegression()


class QDA(AbstractModel):
    def __init__(self):
        super().__init__()
        self.model = QuadraticDiscriminantAnalysis()

    def evaluate_f1(self, X_test, y_test):
        qda_pred = self.model.predict(X_test)
        qda_pred = np.where(np.array(qda_pred) <= 0.5, 0, 1)
        qda_score = f1_score(y_test, qda_pred)
        return np.mean(qda_score)

    def roc_curve(self, X_test, y_test):
        # Obtén las probabilidades de la clase positiva
        y_scores = self.model.predict_proba(X_test)[:, 1]
        # Calcula la curva ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_scores)
        return fpr, tpr, thresholds


class KNN(AbstractModel):
    def __init__(self, n_neighbors=5, weights="uniform", metric="minkowski"):
        super().__init__()
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors, weights=weights, metric=metric
        )


class SVM(AbstractModel):
    def __init__(self, C=1, kernel="linear", random_state=None, probability=False):
        super().__init__()
        self.model = SVC(
            kernel=kernel, C=C, random_state=random_state, probability=probability
        )


class MLP(AbstractModel):
    def __init__(self, hls=(10,), activation="relu", max_iter=1500):
        super().__init__()
        self.model = MLPClassifier(
            hidden_layer_sizes=hls,
            activation=activation,
            max_iter=max_iter,
        )


class KMeans(AbstractModel):
    def __init__(self, n_clusters=2):
        super().__init__()
        self.model = SklearnKMeans(n_clusters=n_clusters)
