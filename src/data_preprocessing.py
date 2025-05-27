"""
Módulo de preprocesamiento de datos para análisis de tipo de cambio DOP/USD
Autor: Johan Manuel Tapia Bueno
Fecha: 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_exchange_rate_data(filepath: str) -> pd.DataFrame:
    """
    Carga datos de tipo de cambio desde archivo CSV
    
    Parameters:
    -----------
    filepath : str
        Ruta al archivo CSV con datos históricos
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con fecha como índice y columnas de precio
    """
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """
    Maneja valores faltantes en series temporales
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame con posibles valores faltantes
    method : str
        Método de interpolación ('linear', 'ffill', 'bfill')
        
    Returns:
    --------
    pd.DataFrame
        DataFrame sin valores faltantes
    """
    df_clean = df.copy()
    
    if method == 'linear':
        df_clean = df_clean.interpolate(method='linear')
    elif method == 'ffill':
        df_clean = df_clean.fillna(method='ffill')
    elif method == 'bfill':
        df_clean = df_clean.fillna(method='bfill')
    
    # Eliminar filas restantes con NaN
    df_clean = df_clean.dropna()
    
    return df_clean


def check_stationarity(timeseries: pd.Series, significance_level: float = 0.05) -> dict:
    """
    Realiza prueba ADF para verificar estacionariedad
    
    Parameters:
    -----------
    timeseries : pd.Series
        Serie temporal a evaluar
    significance_level : float
        Nivel de significancia para la prueba
        
    Returns:
    --------
    dict
        Resultados de la prueba ADF
    """
    from statsmodels.tsa.stattools import adfuller
    
    result = adfuller(timeseries, autolag='AIC')
    
    return {
        'ADF_Statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'observations': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < significance_level
    }


def prepare_data_for_modeling(df: pd.DataFrame, 
                            target_col: str = 'Close',
                            train_size: float = 0.8) -> Tuple[pd.Series, pd.Series]:
    """
    Prepara datos para modelado dividiendo en train/test
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame completo
    target_col : str
        Columna objetivo
    train_size : float
        Proporción de datos para entrenamiento
        
    Returns:
    --------
    Tuple[pd.Series, pd.Series]
        Series de entrenamiento y prueba
    """
    n = len(df)
    train_size = int(n * train_size)
    
    train = df[target_col][:train_size]
    test = df[target_col][train_size:]
    
    print(f"Tamaño conjunto entrenamiento: {len(train)} observaciones")
    print(f"Tamaño conjunto prueba: {len(test)} observaciones")
    print(f"Período entrenamiento: {train.index[0]} a {train.index[-1]}")
    print(f"Período prueba: {test.index[0]} a {test.index[-1]}")
    
    return train, test


def create_exogenous_variables(start_date: str, 
                             end_date: str,
                             freq: str = 'D') -> pd.DataFrame:
    """
    Crea variables exógenas para modelo SARIMAX
    
    Parameters:
    -----------
    start_date : str
        Fecha inicial
    end_date : str
        Fecha final
    freq : str
        Frecuencia de los datos
        
    Returns:
    --------
    pd.DataFrame
        DataFrame con variables exógenas
    """
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Crear DataFrame vacío
    exog_vars = pd.DataFrame(index=dates)
    
    # Agregar variables dummy para días de la semana
    exog_vars['day_of_week'] = exog_vars.index.dayofweek
    exog_vars['is_monday'] = (exog_vars['day_of_week'] == 0).astype(int)
    exog_vars['is_friday'] = (exog_vars['day_of_week'] == 4).astype(int)
    
    # Agregar variables dummy para meses
    exog_vars['month'] = exog_vars.index.month
    exog_vars['is_december'] = (exog_vars['month'] == 12).astype(int)
    exog_vars['is_january'] = (exog_vars['month'] == 1).astype(int)
    
    # Variable para capturar tendencia
    exog_vars['trend'] = np.arange(len(exog_vars))
    
    return exog_vars


if __name__ == "__main__":
    # Ejemplo de uso
    print("Módulo de preprocesamiento de datos cargado correctamente")
