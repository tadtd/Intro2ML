import numpy as np
import util

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)
    mse_list = []
    model = LocallyWeightedLinearRegression(tau=0.5)
    model.fit(x_train, y_train)
    for tau in tau_values:
        model.tau = tau
        y_pred = model.predict(x_val)
        mse = np.mean((y_pred - y_val)**2)
        mse_list.append(mse)

        util.plt.figure()
        util.plt.title(f'tau={tau}')
        util.plt.plot(x_train, y_train, 'bx', linewidth=1)
        util.plt.plot(x_val, y_pred, 'ro', linewidth=1)
        util.plt.xlabel('x')
        util.plt.ylabel('y')
        util.plt.savefig(f'output/p05_c_{tau}.png')
    
    best_tau = tau_values[np.argmin(mse_list)]
    print(f'Validation set: Lowest MSE={min(mse_list)}, tau={best_tau}')
    model.tau = best_tau

    x_test, y_test = util.load_dataset(test_path, add_intercept=True)

    y_pred = model.predict(x_test)
    np.savetxt(pred_path, y_pred)
    mse = np.mean((y_pred - y_test)**2)
    print(f'Test set: tau={best_tau}, MSE={mse}')
    # Search tau_values for the best tau (lowest MSE on the validation set)
    # Fit a LWR model with the best tau value
    # Run on the test set to get the MSE value
    # Save predictions to pred_path
    # Plot data
    # *** END CODE HERE ***
