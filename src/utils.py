import os
import sys
import dill

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a Python object to a specified file path using the dill module.

    This function serializes the provided object and writes it to a file in binary mode. 
    It also ensures that the directory path for the file exists, creating any necessary 
    directories if they don't already exist.

    Args:
        file_path (str): The path (including file name) where the object should be saved.
        obj (object): The Python object to be saved (e.g., a model, preprocessor, etc.).

    Raises:
        CustomException: If there is any error during the process of saving the object, 
                         a custom exception is raised with the error message and traceback.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, parameters):
    """
    Evaluate multiple regression models based on R² score, with hyperparameter tuning.
    
    This function evaluates each model provided in the `models` dictionary using cross-validation 
    for hyperparameter tuning via `GridSearchCV`, then trains the model using the best hyperparameters 
    and evaluates the model on the test set.

    Args:
        X_train (np.array): Training feature data
        y_train (np.array): Training target data
        X_test (np.array): Testing feature data
        y_test (np.array): Testing target data
        models (dict): A dictionary where keys are model names and values are model instances
        parameters (dict): A dictionary where keys are model names and values are dictionaries of hyperparameters 
                           to tune for each model

    Returns:
        dict: A dictionary with model names as keys and their corresponding R² scores on the test data
    """
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
            parameter = parameters[list(models.keys())[i]]
            
            # Hyperparameter Tuning with GridSearchCV
            gs = GridSearchCV(model, parameter, cv=3)
            gs.fit(X_train, y_train)
            
            # Set the model to the best hyperparameters found
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            
            # model.fit(X_train, y_train)  # Training the model 
            
            # y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
            
    except Exception as e:
        raise CustomException(e,sys)