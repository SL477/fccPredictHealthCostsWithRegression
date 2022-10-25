from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
import matplotlib.pyplot as plt


def regression_stats(y_true: list, y_pred: list) -> None:
    """Get the RMSE, MSE and graphs

    Parameters
    ----------
    y_true: numpy array
        The actual data

    y_pred: numpy array
        The predicted data"""
    print('RMSE:', mean_squared_error(y_true, y_pred, squared=False))
    print('MAE:', mean_absolute_error(y_true, y_pred))
    print('R2:', r2_score(y_true, y_pred))

    # graph the data
    plt.scatter(y_pred, y_true, alpha=0.5)
    plt.axline((0, 0), slope=1)
    plt.title('Predicted versus True')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
