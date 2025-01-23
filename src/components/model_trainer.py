import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        """
        Train and evaluate multiple models to select the best one based on R² score, 
        and save the best model if the score is above a threshold.

        Args:
            train_array (np.array): Training data
            test_array (np.array): Testing data

        Raises:
            CustomException: If no model performs well (R² < 0.6)

        Returns:
            float: The R² score of the best model on the test set
        """
        try:
            logging.info("Splitting the train and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }
            
            models_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            
            # Find the model with the best R² score
            best_model_name, best_r2_score = max(models_report.items(), key=lambda x: x[1])
            best_model = models[best_model_name]
            
            if best_r2_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"The best model was identified as {best_model_name} based on the highest r2_score: {best_r2_score} obtained on the test data")
            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Instead of this:
            # y_predicted = best_model.predict(X_test)
            # r2_score_value = r2_score(y_test, y_predicted) # same as best_r2_score
            # return r2_score_value
            
            # You can simply return:
            return best_r2_score
            
        except Exception as e:
            raise CustomException(e,sys)