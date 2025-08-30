import numpy as np
from collections import Counter
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def find_best_split(feature_vector, target_vector):
    sorted_indices = np.argsort(feature_vector)
    sorted_feature = feature_vector[sorted_indices]
    sorted_target = target_vector[sorted_indices]
    
    counts_left = np.cumsum(sorted_target)
    counts_right = np.sum(sorted_target) - counts_left

    unique_feature, split_indices =  np.unique(sorted_feature, return_index=True)
    split_indices = split_indices[1:] - 1

    thresholds = (unique_feature[1:] + unique_feature[:-1]) / 2.0

    n = target_vector.shape[0]
    L = split_indices + 1
    counts_left = counts_left[split_indices]
    counts_right = counts_right[split_indices]
    R = n - L

    pl = counts_left/ L
    hl = 2 * pl * (1 - pl)
    pr = counts_right/ R
    hr = 2 * pr * (1 - pr)

    ginis = -(L/ n) * hl - (R/ n) * hr
    try:
        best_index = np.argmax(ginis)
    except ValueError:
        return None, None, None, None
    threshold_best, gini_best = thresholds[best_index], ginis[best_index]
    return thresholds, ginis, threshold_best, gini_best


class DecisionTree:
    def __init__(self, feature_types, feature_shape_on_fit=False,
                 max_depth=None, min_samples_split=None, min_samples_leaf=None):

        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")
        
        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

        if self._max_depth:
            self._tree["depth"] = 0

    def _fit_node(self, sub_X, sub_y, node, depth = 0):

        if np.all(sub_y == sub_y[0]):                         
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        #if len(np.unique(sub_y)) == 1 or self._max_depth == 0:
            #node["type"] = "terminal"
            #node["class"] = Counter(sub_y).most_common(1)[0][0]
            #return

        if len(np.unique(sub_y)) == 1 or depth == self._max_depth:
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return
        if (self._min_samples_split and len(sub_X) < self._min_samples_split) or (self._min_samples_leaf and  np.any(np.bincount(sub_y) < self._min_samples_leaf)):
                node["type"] = "terminal"
                node["class"] = Counter(sub_y).most_common(1)[0][0]
                return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):              
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count 

                sorted_categories = list(map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
               
                categories_map = dict(zip(sorted_categories, list(range(len(sorted_categories)))))
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError

            if feature_vector is None or len(np.unique(feature_vector)) == 1:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if threshold is not None:
                if gini_best is None or gini > gini_best:
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector < threshold

                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0],
                                                  filter(lambda x: x[1] < threshold, categories_map.items())))
                    else:
                        raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]  
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError

        node["left_child"], node["right_child"] = {}, {}

        if self._max_depth:
            node["left_child"]["depth"] = node["depth"] + 1
            node["right_child"]["depth"] = node["depth"] + 1

        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["class"]
    
        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[node["feature_split"]] in node["categories_split"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)


class LinearRegressionTree():
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None, loss="mse", n_quantiles=10):
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf
        self._loss = loss 
        self._n_quantiles = n_quantiles
        self._tree = {}

    def _split_data(self, X, y, feature, threshold):
        if self._feature_types[feature] == "real":
            mask = X[:, feature] < threshold
        else:
            mask = np.isin(X[:, feature], threshold)
        return X[mask], y[mask], X[~mask], y[~mask]

    def _fit_node(self, X, y, node, depth=0):
        if len(X) < self._min_samples_split or depth == self._max_depth or len(np.unique(y)) == 1:
            
            node["type"] = "terminal"
            model = LinearRegression().fit(X, y)
            node["model"] = model
            return

        best_split = None
        best_loss = float("inf")

        for feature in range(X.shape[1]):
            thresholds = np.quantile(X[:, feature], np.linspace(0, 1, self._n_quantiles + 1)[1:-1])
            for threshold in thresholds:
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature, threshold)

                if len(y_left) < self._min_samples_leaf or len(y_right) < self._min_samples_leaf:
                    continue

                model_left = LinearRegression().fit(X_left, y_left)
                model_right = LinearRegression().fit(X_right, y_right)

                if self._loss == "mse":
                    loss_left = mean_squared_error(y_left, model_left.predict(X_left))
                    loss_right = mean_squared_error(y_right, model_right.predict(X_right))
                else:
                    raise ValueError("Unknown loss function")

                loss = (len(y_left) / len(y)) * loss_left + (len(y_right) / len(y)) * loss_right

                if loss < best_loss:
                    best_loss = loss
                    best_split = (feature, threshold, X_left, y_left, X_right, y_right)

        if best_split is None:
            
            node["type"] = "terminal"
            model = LinearRegression().fit(X, y)
            node["model"] = model
            return

        feature, threshold, X_left, y_left, X_right, y_right = best_split
        node["type"] = "nonterminal"
        node["feature_split"] = feature
        node["threshold"] = threshold
        node["left_child"] = {}
        node["right_child"] = {}

        self._fit_node(X_left, y_left, node["left_child"], depth + 1)
        self._fit_node(X_right, y_right, node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node["type"] == "terminal":
            return node["model"].predict(x.reshape(1, -1))[0]

        if self._feature_types[node["feature_split"]] == "real":
            if x[node["feature_split"]] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])
        else:
            if x[node["feature_split"]] in node["threshold"]:
                return self._predict_node(x, node["left_child"])
            else:
                return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        return np.array([self._predict_node(x, self._tree) for x in X])