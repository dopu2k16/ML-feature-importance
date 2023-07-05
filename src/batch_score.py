import csv
import os
import pickle

from numpy import mean, std

from src.evaluate import evaluate_model_predict
from src.constants import categorical_features, numerical_features
from src.utils import plot_partial_dependence, plot_permutation_importances, shap_analysis, plot_feature_importances


def batch_prediction(x_train, y_train, x_test, y_test, models):
    """
    Finding the predictions for each model on the dataset by calling the evaluate_model()
    """
    #  the models and store results
    results, names = list(), list()
    # test prediction list for all models
    test_preds = []
    # directory for storing the test predictions
    pred_dir = '../results'
    # creating the results directory
    try:
        os.makedirs(pred_dir)
        print("folder '{}' created ".format(pred_dir))
    except FileExistsError:
        print("folder {} already exists".format(pred_dir))

    for name, model in models.items():
        # prediction and evaluation scores for each model
        scores, y_test_pred, mse_test, r2_test = evaluate_model_predict(model, x_train, y_train,
                                                                        x_test, y_test)
        # appending the test prediction of the respective model
        test_preds.append((model, y_test_pred))
        print("Saving predictions")

        # writing the ground truth label and test prediction into file
        with open(pred_dir + "/" + 'pred_' + f'{name}.txt', "w") as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerows(zip(y_test, y_test_pred))

        # appending the evaluation scores
        results.append(scores)
        # appending the names of the model
        names.append(name)
        # printing the training MSE
        print(name)
        print('Average Training Mean Squared Error for %s is %.3f (%.3f)'
              % (name, mean(abs(scores['train_neg_mean_squared_error'])),
                 std(abs(scores['train_neg_mean_squared_error']))))

        # printing the validation MSE
        print('Average Validation MSE for %s is %.3f (%.3f)'
              % (name, mean(abs(scores['test_neg_mean_squared_error'])),
                 std(abs(scores['test_neg_mean_squared_error']))))
        # printing thr training R^2
        print('Average Training R^2 score for %s is %.3f (%.3f)'
              % (name, mean(scores['train_r2']),
                 std(scores['train_r2'])))
        # printing the validation R^2
        print('Average Validation R^2 score for %s is %.3f (%.3f)'
              % (name, mean(scores['test_r2']),
                 std(scores['test_r2'])))
        # printing the test MSE of a model
        print('Test MSE for %s is %.3f' % (name, mse_test))
        # printing the test R^2 of a model
        print('Test R^2 for %s is %f' % (name, r2_test))

        print('-------------------------------------------------')

        print('-----------Partial Dependency Plots---------------')

        plot_partial_dependence(model, x_train, categorical_features, numerical_features, name)

        print('------------Tree Feature Importance Plot---------------------------------')

        plot_feature_importances(model, name)

        print('-------------Permutation Test importance Plot----------------------------------')

        plot_permutation_importances(model, x_train, y_train, x_test, y_test)

        print('------------------SHAP ANALYSIS PLOTS-------------------------------')

        shap_analysis(model, x_train, x_test)

        print('-------------------------------------------------')

        model_dir = '../models'
        # save the model to disk
        try:
            os.makedirs(model_dir)
            print("folder '{}' created ".format(model_dir))
        except FileExistsError:
            print("folder {} already exists".format(model_dir))
        filename = f'{model_dir}/{name}.pkl'
        pickle.dump(model, open(filename, 'wb'))
        # saving the input features of the model
        model_columns = list(x_train.columns)
        with open(f'{model_dir}/{name}_inp_features.pkl', 'wb') as file:
            pickle.dump(model_columns, file)
