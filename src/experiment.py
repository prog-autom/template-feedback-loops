import numpy as np
from matplotlib import pyplot as plt
import seaborn as sb

from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.datasets import make_regression

import random

from mathmodel import AdditiveLossModel as ALM


def init_random_state(seed):
    np.random.seed(int(seed))
    random.seed = int(seed)
    return seed


def ridge_model(**kwargs):
    """
    Creates a pipeline of a data scaler and a ridge regression.
    Scaler transforms data to zero mean and unit variance.
    Hyperparameters or the regression are tuned via cross-validation

    Uses `StandardScaler` and `RidgeCV` from `scikit-learn`
    :param kwargs: parameters for `RidgeCV`
    :return: an instance of `Pipeline`
    """
    return Pipeline([['scaler', StandardScaler()], ['ridgecv', RidgeCV(**kwargs)]])

def get_synthetic_dataset(**kwargs):
    X, y = make_regression(**kwargs)
    return X, y, {'X': X, 'y': y, 'name': 'make_regression', 'args': kwargs}


def single_model_experiment(X, y, model, model_name="model", train_size=0.9, use_log=False):
    """
    Trains a `model` on the given dataset `X` with the target `y`.

    Saves regression plot to `{model_name}-train-test.png`

    :param X: dataset to train and test on (uses split via `train_size`)
    :param y: target variable for regression
    :param model: a class/constructor of the model, should be `callable` which returns and instance of the model with a `fit` method
    :param model_name: a filename for the model, may include path
    :param seed:
    :param train_size:
    :return: None
    """

    X_train, X_test, \
    y_train, y_test = model_selection.train_test_split(X, y, train_size=train_size)

    if use_log:
        y_train = np.log(y_train)

    # Fit regression model
    gbr = model()
    gbr.fit(X_train, y_train)
    gbr_test = gbr.predict(X_test)
    gbr_train = gbr.predict(X_train)

    if use_log:
        gbr_test = np.exp(gbr_test)
        gbr_train = np.exp(gbr_train)

    print('train: ', np.float16(mean_squared_error(y_train, gbr_train)))
    print('test:  ', np.float16(mean_squared_error(y_test, gbr_test)))

    plt.figure()
    plt.title = "Regression plot"
    sb.regplot(x=y_train, y=gbr_train, label="Train")
    sb.regplot(x=y_test, y=gbr_test, label="Test")
    plt.legend()
    plt.xlabel('y')
    plt.ylabel('prediction')
    plt.savefig(f"{model_name}-train-test.png")

class HiddenLoopExperiment:
    """
    The main experiment for hidden loops paper
    See details in the paper.

    In short.

    Creates a feedback loop on a regression problem (e.g. Boston housing).
    Some of the model predictions are adhered to by users and fed back into the model as training data.
    Users add a normally distributed noise to the log of the target variable (price).
    Uses a sliding window to retrain the model on new data.

    """

    default_state = {
        'p0': 'p0, minimum usage for loop',
        'mse': 'MSE, dynamic_data',
        'Linf': 'Estimated minimum loss',
    }

    default_figures = {
        'Loss dynamics': ['mse', 'Linf']
    }

    def __init__(self, X, y, model, model_name="model"):
        """
        Creates an instance of the experiment

        :param X: a dataset for regression
        :param y: target variable
        :param model: a class/constructor of the model, should be `callable` which returns and instance of the model with a `fit` method
        :param model_name: a filename to use for figures
        """
        self.X = X
        self.y = y
        self.model = model
        self.model_name = model_name

    def prepare_data(self, use_log=False, train_size=0.3, A=0.2):
        """
        Initializes the experiment

        :param train_size: size of the sliding window as a portion of the dataset
        :return: None
        """
        self.use_log = bool(use_log)
        self.train_size = float(train_size)
        self.A = float(A)

        self.X_orig, self.X_new, self.y_orig, self.y_new = \
            model_selection.train_test_split(
                self.X,
                np.log(self.y) if self.use_log else self.y,
                train_size=self.train_size)

        self.train_len = len(self.X_orig)

        self.X_new, self.X_orig_tst, self.y_new, self.y_orig_tst = \
            model_selection.train_test_split(
                self.X_new,
                self.y_new,
                test_size=int(0.25*len(self.X_orig)))

        self.X_curr = self.X_orig
        self.y_curr = self.y_orig
        self.index = []
        self.trace = []
        self.mse = []
        self.mse_new = []
        self.m2, self.m2_orig = None, None
        self.p0 = []
        self.Linf = []

    def _get_z_k(self, X, y, **model_params):
        usage = model_params['usage']
        adherence = model_params['adherence']
        if np.random.random() <= float(usage):
            pred = self.model_instance.predict([X])
            new_price = np.random.normal(pred, np.sqrt(self.m2) * float(adherence))[0]
        else:
            new_price = y
        return new_price

    def _add_instances(self, X, y, **model_params):
        """
        This is a generator function (co-routine) for the sliding window loop.
        Works as follows.

        Called once when the loop is initialized.
        Python creates a generator that returns any values provided from this method.
        The method returns the next value via `yield` and continues when `next()` is called on the generator.

        `X` and `y` are set on the first invocation.

        :param X:
        :param y:
        :param usage: how closely users adhere to predictions: `0` means exactly
        :param adherence: share of users to follow prediction
        :return: yields a new sample index from `X`, new price - from `y` or as model predicted
        """

        for sample in np.random.permutation(len(X)):
            pred = self._get_z_k(X[sample], y[sample], **model_params)
            self.X_curr = np.concatenate((self.X_curr[1:], [self.X_new[sample]]))
            self.y_curr = np.concatenate((self.y_curr[1:], [pred]))

            yield sample, pred

    def eval_math(self, L0, Linf=None, **modelargs):
        Linf_v = ALM.get_L_inf(L0, adherence=modelargs['adherence'],
                                usage=modelargs['usage'])

        if Linf is not None:
            Linf.append(Linf_v)

        return Linf_v

    def eval_metrics(self, model, X, y, mse=None):
        pred = model.predict(X)

        mse_v = mean_squared_error(y, pred)

        if mse is not None:
            mse.append(mse_v)
        return mse_v


    def run_experiment(self, **params):
        """
        Main method of the experiment

        :param adherence: how closely users follow model predictions
        :param usage: how often users follow predictions
        :param step: number of steps the model is retrained
        :return: None
        """

        metrics = dict(mse=self.mse)
        metrics_math = dict(Linf=self.Linf)

        def save_iter(iter):
            self.index.append(iter)
            self.m2 = self.eval_metrics(self.model_instance, self.X_tst, self.y_tst, **metrics)

            if self.m2_orig is None:
                self.m2_orig = self.m2

            self.trace.append(dict(
                m2=self.m2,
                Linf=self.eval_math(L0=self.m2_orig, k=0, **metrics_math, **modelargs))
            )

        params = {k:float(params[k]) for k in params.keys()}
        modelargs = dict(train_size=int(self.train_len),
                         A=self.A)
        modelargs.update(params)

        self.X_tr, self.X_tst, self.y_tr, self.y_tst = model_selection.train_test_split(self.X_curr, self.y_curr)

        self.model_instance = self.model()
        self.model_instance.fit(self.X_tr, self.y_tr)

        save_iter(0)

        self.m2_orig = self.m2
        self.p0 = [ALM.get_p_0(adherence=modelargs['adherence'],
                              A=modelargs['A'])]

        i = 0
        for idx, pred in self._add_instances(self.X_new,
                                             self.y_new,
                                             **params):
            i += 1

            if i % int(params['step']) == 0:
                self.X_tr, self.X_tst, \
                self.y_tr, self.y_tst = model_selection.train_test_split(self.X_curr, self.y_curr)

                self.model_instance = self.model()
                self.model_instance.fit(self.X_tr, self.y_tr)

                save_iter(self.index[-1] + 1)
