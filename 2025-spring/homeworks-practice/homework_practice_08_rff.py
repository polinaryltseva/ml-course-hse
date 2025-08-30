import numpy as np

from typing import Callable

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureCreatorPlaceholder(BaseEstimator, TransformerMixin):
    def __init__(self, n_features, new_dim, func: Callable = np.cos):
        self.n_features = n_features
        self.new_dim = new_dim
        self.w = None
        self.b = None
        self.func = func

    def fit(self, X, y=None):
        
        return self

    def transform(self, X, y=None):
        return X


class RandomFeatureCreator(FeatureCreatorPlaceholder):
    def fit(self, X, y=None):
        n_samples = X.shape[0]
        n_pairs = min(10000, n_samples)
        distances = []
        d = X.shape[1]
        
        while len(distances) < n_pairs:
            idx1 = np.random.randint(0, n_samples)
            idx2 = np.random.randint(0, n_samples)
            
            if idx1 != idx2:
                dist = np.sum((X[idx1] - X[idx2]) ** 2)
                distances.append(dist)
        
        sigma_squared = np.median(np.array(distances))
        self.w = np.random.normal(0, 1 / np.sqrt(sigma_squared + 1e-8), (self.n_features, d))
        self.b = np.random.uniform(-np.pi, np.pi, self.n_features)
        return self

    def transform(self, X, y=None):
        return np.cos(np.dot(X, self.w.T) + self.b)
    



class OrthogonalRandomFeatureCreator(RandomFeatureCreator):
    def fit(self, X, y=None):
        raise NotImplementedError


class RFFPipeline(BaseEstimator):
    """
    Пайплайн, делающий последовательно три шага:
        1. Применение PCA
        2. Применение RFF
        3. Применение классификатора
    """
    def __init__(
            self,
            n_features: int = 1000,
            new_dim: int = 50,
            use_PCA: bool = True,
            feature_creator_class=FeatureCreatorPlaceholder,
            classifier_class=LogisticRegression,
            classifier_params=None,
            func=np.cos,
    ):
        """
        :param n_features: Количество признаков, генерируемых RFF
        :param new_dim: Количество признаков, до которых сжимает PCA
        :param use_PCA: Использовать ли PCA
        :param feature_creator_class: Класс, создающий признаки, по умолчанию заглушка
        :param classifier_class: Класс классификатора
        :param classifier_params: Параметры, которыми инициализируется классификатор
        :param func: Функция, которую получает feature_creator при инициализации.
                     Если не хотите, можете не использовать этот параметр.
        """
        self.n_features = n_features
        self.new_dim = new_dim
        self.use_PCA = use_PCA
        #if classifier_params is None:
            #lassifier_params = {'max_iter': 1100}  
        #else:
            #classifier_params = {**{'max_iter': 1100}, **classifier_params}
        if classifier_params is None:
            classifier_params = {}
        if classifier_class == LinearSVC:
            self.classifier = LinearSVC(**classifier_params)
        else:
            self.classifier = LogisticRegression(**classifier_params)
            
        self.feature_creator = feature_creator_class(
            n_features=self.n_features, new_dim=self.new_dim, func=func
        )
        
        pipeline_steps = []
        if self.use_PCA:
            pipeline_steps.append(('pca', PCA(n_components=self.new_dim)))
        else:
            pipeline_steps.append(('scaler', StandardScaler()))
        pipeline_steps.append(('rff', self.feature_creator))
        pipeline_steps.append(('classifier', self.classifier))
        
        self.pipeline = Pipeline(pipeline_steps)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)

    def predict(self, X):
        return self.pipeline.predict(X)

    def score(self, X, y):
        return self.pipeline.score(X, y)
