from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer

from typing import Optional
import copy
import time

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
        self,
        base_model_class = DecisionTreeRegressor,
        base_model_params: Optional[dict] = None,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        early_stopping_rounds = None,
        eval_set = None,
        subsample: float = 1.0,
        bagging_temperature: float = 1.0,
        bootstrap_type: str = 'Bernoulli',
        bootstrap = False,
        rsm=1.0, 
        quantization_type=None, 
        nbins=255
    ):
        self.base_model_class = base_model_class
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.early_stopping_rounds: int = early_stopping_rounds  

        if eval_set is None:
            self.eval_set = None
        else:
            self.eval_set: list = eval_set  

        self.models: list = []
        self.gammas: list = []
        self.depths: list = []

        self.subsample = subsample
        self.bagging_temperature = bagging_temperature
        self.bootstrap_type = bootstrap_type
        self.bootstrap = bootstrap

        self.rsm = rsm 
        self.quantization_type = quantization_type  
        self.nbins = nbins 

        self.learning_rate: float = learning_rate

        self.val_losses = []

        self.history = defaultdict(list) # {"train_roc_auc": [], "train_loss": [], ...}

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)
        #self.loss_derivative = lambda y, z: z * np.exp(np.log(y))  # Ð˜ÑÐ¿Ñ€Ð°Ð²ÑŒÑ‚Ðµ Ñ„Ð¾Ñ€Ð¼ÑƒÐ»Ñƒ Ð½Ð° Ð¿Ñ€Ð°Ð²Ð¸Ð»ÑŒÐ½ÑƒÑŽ. 

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        if self.bootstrap_type == 'Bernoulli':
            indices = np.random.choice(np.arange(n_samples), size=int(self.subsample * n_samples), replace=True)
        return X[indices], y[indices]

    def _quantize_features(self, X):
        if self.quantization_type is None:
            return X

        if self.quantization_type == 'uniform':
            X_dense = X.toarray()
            X_quantized = np.zeros_like(X_dense)
            for i in range(X.shape[1]):
                min_val = np.min(X_dense[:, i])
                max_val = np.max(X_dense[:, i])
                bins = np.linspace(min_val, max_val, self.nbins + 1)
                X_quantized[:, i] = np.digitize(X_dense[:, i], bins[:-1]) - 1
            return X_quantized
        
        elif self.quantization_type == 'quantile':
            X_dense = X.toarray()
            X_quantized = np.zeros_like(X_dense)
            for i in range(X.shape[1]):
                quantiles = np.quantile(X_dense[:, i], np.linspace(0, 1, self.nbins + 1))
                X_quantized[:, i] = np.digitize(X_dense[:, i], quantiles[:-1]) - 1
            return X_quantized
 
    def partial_fit(self, X, y, old_predictions):
        model = self.base_model_class(**self.base_model_params)
        residuals = y - self.sigmoid(old_predictions)
        model.fit(X, residuals)
        self.models.append(model)
        self.depths.append(model.get_depth())
        new_predictions = model.predict(X)
        gamma = self.find_optimal_gamma(y, old_predictions, new_predictions)
        self.gammas.append(gamma)
        return new_predictions

    def _select_features(self, X):

        num_features = X.shape[1]
        if isinstance(self.rsm, float):
          num_selected = int(num_features * self.rsm)
        else:
          num_selected = min(num_features, self.rsm)

        if num_selected == num_features:
          return X

        selected_indices = resample(np.arange(num_features), replace=False, n_samples=num_selected)
        return X[:, selected_indices]

    def fit(self, X_train, y_train, X_val=None, y_val=None, plot=False):
        """
        :param X_train: features array (train set)
        :param y_train: targets array (train set)
        :param X_val: features array (eval set)
        :param y_val: targets array (eval set)
        :param plot: bool 
        """
        train_predictions = np.zeros(y_train.shape[0])

        best_val_score = -np.inf
        rounds_without_improvement = 0

        for i in range(self.n_estimators):

            if self.bootstrap:
                X_sample, y_sample = self.bootstrap_sample(X_train, y_train)
                if train_predictions.shape[0] != y_sample.shape[0]:
                    train_predictions = np.zeros(y_sample.shape[0])
            else:
                X_sample, y_sample = X_train, y_train
            
            X_selected = self._select_features(X_sample)
            X_sample = self._quantize_features(X_selected)
            
            new_predictions = self.partial_fit(X_sample, y_sample, train_predictions)
            train_predictions += self.learning_rate * self.gammas[-1] * new_predictions
            train_loss = self.loss_fn(y_sample, train_predictions)
            train_auc = roc_auc_score(y_sample, self.sigmoid(train_predictions))

            self.history['train_loss'].append(train_loss)
            self.history['train_roc_auc'].append(train_auc)

            if self.eval_set is not None:
               
                X_val, y_val = self.eval_set
                val_predictions = self.predict_proba(X_val)
                val_auc = roc_auc_score(y_val, val_predictions[:, 1])  
                self.history['val_roc_auc'].append(val_auc)

                if val_auc > best_val_score:
                    best_val_score = val_auc
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1

                if rounds_without_improvement >= self.early_stopping_rounds:
                    print(f"Early stopping after {i+1} iterations.")
                    break
            

        if plot:
            self.plot_history(X_train, y_train)



    def predict_proba(self, X):
     
        total_predictions = np.zeros(X.shape[0])
        for model, gamma in zip(self.models, self.gammas):
            total_predictions += gamma * model.predict(X)

        p1 = self.sigmoid(total_predictions)
        p0 = 1 - p1
        return np.column_stack((p0, p1))
        

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, X, y):
        return score(self, X, y)
        
    def plot_history(self, X, y):
        """
        :param X: features array (any set)
        :param y: targets array (any set)
        """
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label="Train loss")

        val_losses = self.val_losses
        
        if self.eval_set is not None:
            X_val = self.eval_set[0]
            y_val = self.eval_set[1]
            val_predictions = np.zeros(y_val.shape[0])
            for i in range(len(self.models)):
                val_predictions += self.learning_rate * self.gammas[i] * self.models[i].predict(X_val)
                loss = self.loss_fn(y_val, val_predictions)
                val_losses.append(loss)

        self.val_losses = val_losses
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", color="orange")
        plt.title("Losses while fitting")
        plt.legend()
        plt.show()

    def feature_importances_(self):
        feature_importances = np.zeros(self.models[0].feature_importances_.shape[0])
        for model in self.models:
            feature_importances += model.feature_importances_
        feature_importances /= 1.0 * len(self.models)
        return feature_importances / np.sum(feature_importances)

    def get_losses(self):
        X_val = self.eval_set[0]
        y_val = self.eval_set[1]
        val_predictions = np.zeros(y_val.shape[0])
        val_losses = []
        for i in range(len(self.models)):
            val_predictions += self.learning_rate * self.gammas[i] * self.models[i].predict(X_val)
            loss = self.loss_fn(y_val, val_predictions)
            val_losses.append(loss)
        return val_losses