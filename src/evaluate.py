from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def evaluate_model_predict(model, x, y, x_test, y_test):
    """
    Performs the cross validation of the ML algorithms and returns the training,
     validation scores, and test predictions.
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    model.fit(x, y)
    # cross validation
    scores = cross_validate(model, x, y, scoring=('neg_mean_squared_error', 'r2'),
                            cv=cv, n_jobs=-1, return_train_score=True)
    # test prediction from the trained ml model
    y_test_pred = model.predict(x_test)

    # test metrics like r^2, mse
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    return scores, y_test_pred, mse_test, r2_test
