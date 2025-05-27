"""
Módulo de modelos de predicción para tipo de cambio DOP/USD
Autor: Johan Manuel Tapia Bueno
Fecha: 2025
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import pmdarima as pm
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class ARIMAModel:
    """Clase para modelo ARIMA con selección automática de parámetros"""
    
    def __init__(self, seasonal: bool = False):
        self.seasonal = seasonal
        self.model = None
        self.order = None
        self.seasonal_order = None
        
    def fit(self, train_data: pd.Series, auto: bool = True) -> None:
        """
        Ajusta modelo ARIMA/SARIMA
        
        Parameters:
        -----------
        train_data : pd.Series
            Serie temporal de entrenamiento
        auto : bool
            Si True, usa auto_arima para selección de parámetros
        """
        if auto:
            self.model = pm.auto_arima(
                train_data,
                seasonal=self.seasonal,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore',
                max_p=5,
                max_q=5,
                max_order=10
            )
            self.order = self.model.order
            if self.seasonal:
                self.seasonal_order = self.model.seasonal_order
        else:
            # Valores por defecto
            self.order = (1, 1, 1)
            if self.seasonal:
                self.seasonal_order = (1, 1, 1, 12)
                self.model = SARIMAX(
                    train_data,
                    order=self.order,
                    seasonal_order=self.seasonal_order
                ).fit(disp=False)
            else:
                self.model = ARIMA(train_data, order=self.order).fit()
                
    def predict(self, steps: int) -> pd.Series:
        """
        Realiza predicciones
        
        Parameters:
        -----------
        steps : int
            Número de pasos a predecir
            
        Returns:
        --------
        pd.Series
            Serie con predicciones
        """
        return self.model.predict(n_periods=steps)
    
    def get_params(self) -> Dict:
        """Retorna parámetros del modelo"""
        params = {
            'order': self.order,
            'aic': self.model.aic() if hasattr(self.model, 'aic') else None,
            'bic': self.model.bic() if hasattr(self.model, 'bic') else None
        }
        if self.seasonal:
            params['seasonal_order'] = self.seasonal_order
        return params


class ProphetModel:
    """Clase para modelo Prophet"""
    
    def __init__(self, 
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False):
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode='multiplicative'
        )
        self.train_data = None
        
    def fit(self, train_data: pd.Series) -> None:
        """
        Ajusta modelo Prophet
        
        Parameters:
        -----------
        train_data : pd.Series
            Serie temporal de entrenamiento
        """
        # Prophet requiere formato específico
        df_prophet = pd.DataFrame({
            'ds': train_data.index,
            'y': train_data.values
        })
        self.train_data = df_prophet
        self.model.fit(df_prophet)
        
    def predict(self, periods: int) -> pd.DataFrame:
        """
        Realiza predicciones
        
        Parameters:
        -----------
        periods : int
            Número de períodos a predecir
            
        Returns:
        --------
        pd.DataFrame
            DataFrame con predicciones y intervalos
        """
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)
        return forecast
    
    def plot_components(self):
        """Grafica componentes del modelo"""
        forecast = self.model.predict(self.train_data)
        self.model.plot_components(forecast)


class SARIMAXModel:
    """Clase para modelo SARIMAX con variables exógenas"""
    
    def __init__(self, order: Tuple, seasonal_order: Tuple):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        
    def fit(self, train_data: pd.Series, exog_data: Optional[pd.DataFrame] = None) -> None:
        """
        Ajusta modelo SARIMAX
        
        Parameters:
        -----------
        train_data : pd.Series
            Serie temporal de entrenamiento
        exog_data : pd.DataFrame, optional
            Variables exógenas
        """
        self.model = SARIMAX(
            train_data,
            exog=exog_data,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)
        
    def predict(self, steps: int, exog_data: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Realiza predicciones
        
        Parameters:
        -----------
        steps : int
            Número de pasos a predecir
        exog_data : pd.DataFrame, optional
            Variables exógenas para predicción
            
        Returns:
        --------
        pd.Series
            Serie con predicciones
        """
        start = len(self.model.data.endog)
        end = start + steps - 1
        return self.model.predict(start=start, end=end, exog=exog_data)
    
    def get_summary(self):
        """Retorna resumen del modelo"""
        return self.model.summary()


if __name__ == "__main__":
    print("Módulo de modelos cargado correctamente")
