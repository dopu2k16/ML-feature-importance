import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.inspection import PartialDependenceDisplay, permutation_importance

import shap

np.bool = np.bool_
np.int = np.int_
np.float = np.float_


def load_data_csv(filename, delimiter):
    """
    Loading the dataset if the file exists.
    """
    try:
        if os.path.isfile(filename):
            df = pd.read_csv(filename, delimiter=delimiter)
            return df
    except FileNotFoundError as e:
        print("File not found", e)


def merge_and_process_data(reservations_file, vehicles_file):
    # Read the reservations and vehicles files into DataFrames

    df_reserve = load_data_csv(reservations_file, delimiter=',')
    df_vehicles = load_data_csv(vehicles_file, delimiter=',')

    # Merge the datasets on vehicle_id
    data = pd.merge(df_vehicles, df_reserve, on='vehicle_id', how='outer')

    # Group by vehicle_id and reservation_type, and count the number of bookings
    reservation_counts = data.groupby(['vehicle_id', 'reservation_type']).size().reset_index(name='num_reservations')

    # Merge the reservation counts with the original data based on 'vehicle_id' and 'reservation_type'
    data = pd.merge(data, reservation_counts, on=['vehicle_id', 'reservation_type'], how='outer')

    # Fill missing values with zeros
    data['num_reservations'].fillna(0, inplace=True)
    data['reservation_type'].fillna(0, inplace=True)

    # Drop duplicates from the dataframe
    data = data.drop_duplicates()

    return data


def split_data(data, numerical_features, categorical_features, target_variable, random_state=42):
    X = data[categorical_features + numerical_features]
    y = data[target_variable]
    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=random_state)
    return x_train, x_test, y_train, y_test


def plot_partial_dependence(tree_pipeline, x_train, categorical_features, numerical_features, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{model_name} (Categorical Features)")
    tree_disp = PartialDependenceDisplay.from_estimator(tree_pipeline, x_train, categorical_features, ax=ax)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(f"{model_name} (Numerical Features)")
    tree_disp = PartialDependenceDisplay.from_estimator(tree_pipeline, x_train, numerical_features, ax=ax)

    plt.show()


def plot_feature_importances(tree_pipeline, model_name):
    feature_names = tree_pipeline[:-1].get_feature_names_out()
    mdi_importances = pd.Series(
        tree_pipeline[-1].feature_importances_, index=feature_names
    ).sort_values(ascending=True)

    ax = mdi_importances.plot.barh()
    ax.set_title(f"{model_name} Feature Importances (MDI)")
    ax.figure.tight_layout()
    plt.show()


def plot_permutation_importances(tree_pipeline, x_train, y_train, x_test, y_test):
    train_result = permutation_importance(
        tree_pipeline, x_train, y_train, n_repeats=10, random_state=42, n_jobs=2
    )
    test_results = permutation_importance(
        tree_pipeline, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2
    )
    sorted_importances_idx = train_result.importances_mean.argsort()

    train_importances = pd.DataFrame(
        train_result.importances[sorted_importances_idx].T,
        columns=x_train.columns[sorted_importances_idx],
    )
    test_importances = pd.DataFrame(
        test_results.importances[sorted_importances_idx].T,
        columns=x_train.columns[sorted_importances_idx],
    )

    for name, importances in zip(["train", "test"], [train_importances, test_importances]):
        ax = importances.plot.box(vert=False, whis=10)
        ax.set_title(f"Permutation Importances ({name} set)")
        ax.set_xlabel("Decrease in performance")
        ax.axvline(x=0, color="k", linestyle="--")
        ax.figure.tight_layout()
        plt.show()


def shap_analysis(tree_pipeline, x_train, x_test):
    explainer = shap.TreeExplainer(tree_pipeline.named_steps["model"])

    x_tr = tree_pipeline.named_steps['preprocessor'].transform(x_train)
    shap_values_tr = explainer.shap_values(x_tr)

    x_tes = tree_pipeline.named_steps['preprocessor'].transform(x_test)
    shap_values_tes = explainer.shap_values(x_tes)

    print("TREE SHAP ANALYSIS FOR TRAIN SET\n")

    print('------------TREE SHAP Feature Importance Plot-------------------------------------')

    shap.summary_plot(shap_values_tr, x_tr, feature_names=tree_pipeline[:-1].get_feature_names_out(),
                      plot_type='bar')
    print('------------------TREE SHAP Summary Plot-------------------------------')

    shap.summary_plot(shap_values_tr, x_tr, feature_names=tree_pipeline[:-1].get_feature_names_out())

    print('-------------TREE SHAP Dependence Plots by top 5 rank------------------------------------')

    for i in range(5):
        shap.dependence_plot(f"rank({i})", shap_values_tr, x_tr,
                             feature_names=tree_pipeline[:-1].get_feature_names_out())
        plt.show()

    print("SHAP ANALYSIS FOR TEST SET\n")

    print('------------TREE SHAP Feature Importance Plot-------------------------------------')

    shap.summary_plot(shap_values_tes, x_tes, feature_names=tree_pipeline[:-1].get_feature_names_out(),
                      plot_type='bar')

    print('------------TREE SHAP Summary Plot-------------------------------------')

    shap.summary_plot(shap_values_tes, x_tes, feature_names=tree_pipeline[:-1].get_feature_names_out())

    print('-------------TREE SHAP Dependence Plots by top 5 rank------------------------------------')

    for i in range(5):
        shap.dependence_plot(f"rank({i})", shap_values_tes, x_tes,
                             feature_names=tree_pipeline[:-1].get_feature_names_out())
        plt.show()
