from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type

import numpy as np

DELTA = 1.2

@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        if self.loss_function is LossFunction.MSE:
            return 2 * x.T @ (self.predict(x) - y) / (y.shape[0])
        if self.loss_function is LossFunction.LogCosh:
            return x.T @ np.tanh(self.predict(x) - y) / (y.shape[0])
        if self.loss_function is LossFunction.MAE:
            return x.T @ np.sign(self.predict(x) - y) / y.shape[0]
        

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        if self.loss_function is LossFunction.MSE:
            return np.mean((self.predict(x) - y)**2)
        if self.loss_function is LossFunction.LogCosh:
            return np.mean(np.log(np.cosh(self.predict(x) - y)))
        if self.loss_function is LossFunction.MAE:
            return np.mean(np.abs(self.predict(x) - y))


    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w 


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        diff = -self.lr() * gradient
        self.w += diff
        return diff

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        loss = self.predict(x) - y
        #if self.loss_function is LossFunction.MSE:
           #return 2 * x.T @ loss / y.shape[0]
        #elif self.loss_function is LossFunction.LogCosh:
            #return x.T @ np.tanh(loss) / x.shape[0]
        return super().calc_gradient(x, y)


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        index = np.random.randint(y.shape[0], size=self.batch_size)
        x_batch = x[index]
        y_batch = np.array(y)[index]
        #w = self.w[self.batch_size]
        loss = x_batch @ self.w - y_batch
        #if self.loss_function is LossFunction.MSE:
            #return 2 * x_batch.T @ loss/ self.batch_size
        #elif self.loss_function is LossFunction.LogCosh:
            #return x.T @ np.tanh(loss) / self.batch_size
        return super().calc_gradient(x_batch, y_batch)



class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights with respect to gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.alpha * self.h + self.lr() * gradient
        self.w -= self.h
        return -self.h


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        m_new = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        v_new = self.beta_2 * self.v + (1 - self.beta_2) * (gradient ** 2)
        self.m = m_new
        self.v = v_new
        self.iteration += 1
        m_k = self.m / (1 - self.beta_1**self.iteration)
        v_k = self.v / (1 - self.beta_2**self.iteration)
        step = -self.lr() * m_k / (np.sqrt(v_k) + self.eps)
        self.w += step
        return step

class AdaMax(VanillaGradientDescent):
    """
    AdaMax gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.LogCosh):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8
        self.u: float = 0.0

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.9

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update weights & params
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        self.v = np.maximum(self.beta_2 * self.v, np.abs(gradient))
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        cur_lr = self.m/(1 - self.beta_1**self.iteration)
        step = -self.lr() * cur_lr/(self.v + self.eps)
        self.w += step
        return step



class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
        l2_gradient: np.ndarray = np.zeros_like(x.shape[1])  # TODO: replace with L2 gradient calculation
        w = self.w
        w[-1] = 0
        l2_gradient = w
        return super().calc_gradient(x, y) + l2_gradient * self.mu 


class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """

class AdaMaxReg(BaseDescentReg, AdaMax):
    """
    Adaptive gradient algorithm with regularization class
    """


def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescent if not regularized else VanillaGradientDescentReg,
        'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
        'momentum': MomentumDescent if not regularized else MomentumDescentReg,
        'adam': Adam if not regularized else AdamReg, 
        'adamax': AdaMax if not regularized else AdaMaxReg
    }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))