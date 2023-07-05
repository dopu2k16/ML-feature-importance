from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

import warnings

from src.batch_score import batch_prediction
from src.constants import numerical_features, categorical_features, target_variable
from src.utils import split_data, merge_and_process_data

warnings.filterwarnings("ignore")


def get_ml_models():
    """
    The ML training algorithms for the Lead Generator Problem.
    The following ml algorithms were used to predict a given customer as a hot lead or not.
    """
    models = dict()

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), categorical_features)],
        remainder='passthrough'
    )

    # Random Forest Regression
    rf_model = RandomForestRegressor()
    models['RandomForest'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', rf_model)
    ])

    # GBR Regression
    gbr_model = GradientBoostingRegressor()
    models['GBR'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', gbr_model)
    ])

    # DT Regression
    dt_model = DecisionTreeRegressor()
    models['DecisionTree'] = Pipeline([
        ('preprocessor', preprocessor),
        ('model', dt_model)
    ])

    return models


def main():
    data = merge_and_process_data('../data/reservations.csv', '../data/vehicles.csv')
    x_train, x_test, y_train, y_test = split_data(data, numerical_features, categorical_features,
                                                  target_variable, random_state=42)
    # getting all the implemented ml models
    models = get_ml_models()
    # getting the predictions for both the training and testing datasets
    batch_prediction(x_train, y_train, x_test, y_test, models)


if __name__ == "__main__":
    main()
