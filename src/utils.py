import os
import sys
import dill

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