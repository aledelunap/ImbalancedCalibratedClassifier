from sklearn.calibration import CalibratedClassifierCV
from imblearn.under_sampling import RandomUnderSampler
from copy import deepcopy
import pandas as pd
import numpy as np

class ImbalancedCalibratedClassifiers():

    def __init__(
        self, 
        base_estimator=None, 
        method="sigmoid", 
        n_calibrators=1, 
        undersampler=RandomUnderSampler(
            sampling_strategy="all", 
            replacement=True, 
            random_state=42
        )
    ):

        self.n_calibrators = n_calibrators
        self.undersampler = undersampler
        self.calibrators = [CalibratedClassifierCV(deepcopy(base_estimator), method=method, cv="prefit") for _ in range(n_calibrators)]

    def _create_calibration_sets(self, x: pd.DataFrame, y: pd.DataFrame) -> dict:
        """Create the claibration sets for the number of models in the ensemble.

        """

        calibration_sets = {}

        self.undersampler = self.undersampler.fit(x, y)
        
        for n in range(self.n_calibrators):

            # Undersample minority class
            x_under = self.undersampler.fit_resample(x, y)

            # Save the new random sample
            calibration_sets[n] = x_under
        
        return calibration_sets

    def fit(self, x: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit the set of calibration models.

        Args:
            X (pd.DataFrame): Calibration data, the indices must be unique identifers
            y (pd.DataFrame): Calibration target, the indices must be unique identifers
        """

        # Create the calibration sets
        calibration_sets = self._create_calibration_sets(x, y)

        for n in range(self.n_calibrators):
            # Calibrate each model with the respective calibration set
            self.calibrators[n] = self.calibrators[n].fit(calibration_sets[n][0], calibration_sets[n][1])

        return None


    def predict_proba(self, x: pd.DataFrame) -> list:
        """
        Funcion para calcular la probabilidad promedio del ensemble.
        
        Args:
            x (pd.DataFrame): Data del modelo.
            
        Returns:
            list: Predicciones del ensemble.
        """
        
        # Predict probability of first estimator to initialize
        predictions = self.calibrators[0].predict_proba(x)
        
        for n in range(1, self.n_calibrators):
            predictions += self.calibrators[n].predict_proba(x)
        
        return predictions/self.n_calibrators

    def predict(self,  x: pd.DataFrame) -> list:
        """
        Funcion para calcular el score promedio del ensemble.
        
        Args:
            x (pd.DataFrame): Data del modelo.
            
        Returns:
            list: Predicciones del ensemble.
        """
        
        predictions = self.calibrators[0].predict(x)
        
        for n in range(1, self.n_calibrators):
            predictions += self.calibrators[n].predict(x)
        
        return (np.round(predictions/self.calibrators)).astype(int)