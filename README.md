## Machine Learning Feature Importance
### Author: Mitodru Niyogi
What factors in the features drive the target for regression problem? In this project, I show an example with a vehicle
booking dataset, I try to determine what factors in vehicles drive the total number of reservation? 

Features: vehicle_id, technology, actual_price, recommended_price, num_images, street_parked, description, reservation_type



`src` contains the `batch_score.py`, `utils.py`, `constants.py`, `evaluate.py`, `train.py`  source files

[src](src) contains the source code for the tasks.

 [src/batch_score.py](src/batch_score.py) contains the code for calculating the metrics and outputs feature importance
 plots.

[src/utils.py](src/utils.py) contains various methods for data preprocessing, calculating feature importance methods,
SHAP analysis (Summary Plot, Dependency
Plots), Partial Dependency plots, tree-based feature importance, permutation-based features importance methods.

[src/evaluate.py](src/evaluate.py) contains the evaluation code for model evaluation and cross-validation

[src/train.py](src/train.py) has training methods for training tree based ml algorithms and evaluation, prediction
and analysis plots.

`notebooks` contains the 
#### Exploratory Data Analysis [notebooks/EDA./ipynb](notebooks/EDA.ipynb) 

#### Feature importance analysis:  [notebooks/Feature_importance.ipynb](notebooks/Feature_importance.ipynb).

### Instructions
#### Creating python environment

```python -m venv featureimportance```

#### Activating the environment

```source featureimportance/bin/activate```

#### Installing the required packages

```pip install -r requirements.py```

#### Running the Analysis

```python train.py```


## Methods for feature importance

1. Tree-based feature importance (Random Forest, Decision Tree, GradientBoosting regressors) based on mean decrease
in impurity (MDI) were tried out to calculate the feature importance of the dataset.
2. Permutation-based Feature importance was also performed on both training and test sets to find the important features that
drives the number of reservations.
3. SHAP feature importance along with summary plot and dependency plots were utilized to understand how much each feature
drives the final target variable that is number of reservations count.

### Tree-based models
They provide a measure of feature importance based on the mean decrease in impurity (MDI). 
Impurity is quantified by the splitting criterion of the decision trees (Gini, Log Loss or Mean Squared Error). 
However, this method can give high importance to features that may not be predictive on unseen data when the model is
overfitting. Tree-based feature importance based on provides insights into which features are most informative for
making predictions in tree-based models. It helps identify the key variables that contribute significantly to the 
model's performance and can guide feature selection, interpretation, and model refinement.

### Permutation-based Feature Importance:

Permutation-based feature importance is a technique that measures the importance of each feature by shuffling the values
of that feature and observing the effect on the model's performance.
It calculates the decrease in the model's performance (e.g., accuracy, mean squared error) when a feature's values are
randomly permuted while keeping the other features unchanged.
By comparing the decrease in performance to a baseline, such as the original model's performance, it determines the
relative importance of each feature.
Permutation feature importance provides a feature importance score that represents how much the model relies on each
feature for its predictions.
It is a model-agnostic technique and can be applied to any machine learning algorithm.
Permutation-based feature importance, on the other hand, avoids this issue, since it can be computed on 
unseen data.

Furthermore, impurity-based feature importance for trees are strongly biased and favor high cardinality features 
(typically numerical features) over low cardinality features such as binary features or categorical variables with
a small number of possible categories.  Permutation-based feature importances do not exhibit such a bias. 
Additionally, the permutation feature importance may be computed performance metric on the model predictions
and can be used to analyze any model class (not just tree-based models).


### SHAP Values Feature Importance:

SHAP values are a unified measure of feature importance based on cooperative game theory concepts.
SHAP values quantify the contribution of each feature to the prediction for a specific instance by considering all
possible feature combinations.
They provide a way to assign credits or values to features, indicating how much each feature contributes to the
difference between the expected model output and the actual output for a given instance.
SHAP values take into account the interactions and dependencies between features, providing a more nuanced understanding 
of feature importance. They can be used to explain the individual predictions of a model or to calculate aggregate 
feature importance across multiple instances. SHAP values are model-specific and rely on the specific model structure
and algorithm used.


In summary, permutation feature importance evaluates feature importance by measuring the decrease in model 
performance due to feature shuffling, while SHAP values provide a more comprehensive and model-specific assessment of
feature importance by considering the contribution of each feature to the prediction for individual instances. 
Both techniques can be valuable for understanding feature importance, and therefore, we tried out these methods to
understand the important factors driving the number of reservations and how's the car's technology type seem
to affect reservations.
