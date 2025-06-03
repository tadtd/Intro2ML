import numpy as np
import util

from linear_model import LinearModel


def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    model = LocallyWeightedLinearRegression(tau)
    model.fit(x_train, y_train)
    # Get MSE value on the validation set
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_val)
    mse = np.mean((y_pred- y_val)**2)
    print(f'MSE:{mse}')

    # Plot validation predictions on top of training set
    util.plt.figure()
    util.plt.plot(x_train, y_train, 'bx',linewidth=1)
    util.plt.plot(x_val, y_pred, 'ro', linewidth=1)
    util.plt.xlabel('x')
    util.plt.ylabel('y')
    util.plt.savefig('output/p05b.png')
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x = x
        self.y = y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, _ = x.shape
        y_pred = np.zeros(m)

        for i in range(m):
            # Compute squared distance using all features
            # diff = self.x[:, 1:] - x_query[1:]          # (m, d)
            diff = self.x - x[i]
            distance_sq = np.sum(diff ** 2, axis=1)     # (m,)

            # Compute weights
            weights = np.exp(-distance_sq / (2 * self.tau ** 2)) 
            W = np.diag(weights)

            # Weighted normal equation
            XTW = self.x.T @ W
            XTWX = XTW @ self.x
            XTWy = XTW @ self.y

            try:
                theta = np.linalg.solve(XTWX, XTWy)
            except np.linalg.LinAlgError:
                theta = np.linalg.pinv(XTWX) @ XTWy

            y_pred[i] = x[i] @ theta

        return y_pred
        # *** END CODE HERE ***
