import numpy as np
import util

from linear_model import LinearModel

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    model = LogisticRegression(eps=1e-5)
    model.fit(x_train, y_train)

    util.plot(x_train, y_train, model.theta, 'output/p01b_{}.png'.format(pred_path[-5]))

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = model.predict(x_eval)
    np.savetxt(pred_path, y_pred, fmt='%d')
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m, n = x.shape
        self.theta = np.zeros(n)

        #Newton's Method
        while True:
            old_theta = np.copy(self.theta)
            # Compute the hypothesis
            h = 1 / (1 + np.exp(-x.dot(self.theta)))
            # Compute the Hessian matrix
            hessian = x.T.dot(np.diag(h * (1 - h))).dot(x) / m
            # Compute the gradient
            gradient = x.T.dot(h - y) / m
            # Update theta
            self.theta -= np.linalg.inv(hessian).dot(gradient)
            # Check for convergence
            if np.linalg.norm(old_theta - self.theta, ord=1) < self.eps:
                break

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x.dot(self.theta))) > 0.5
        # *** END CODE HERE ***
