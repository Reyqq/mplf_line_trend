import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
from typing import Dict, List, Tuple, Union, Callable, Optional
from tqdm.notebook import tqdm
import logging
pd.set_option('float_format', '{:.2f}'.format)

np.seterr(divide='ignore', invalid='ignore')



def extract_datetime_components(df: pd.DataFrame) -> pd.DataFrame:
    """
    .. :no-index:
    
    Преобразует столбец 'timestamp', извлекает компоненты даты и времени и устанавливает 'timestamp' в качестве индекса.

    Эта функция выполняет следующие операции:
    1. Преобразует столбец 'timestamp' в формат datetime.
    2. Извлекает дату и время из 'timestamp' в отдельные столбцы.
    3. Устанавливает 'timestamp' в качестве индекса датафрейма.

    Args:
        df (pd.DataFrame): Исходный датафрейм с столбцом 'timestamp'.

    Returns:
        pd.DataFrame: Обновленный датафрейм с новыми столбцами 'date' и 'time',
                      и 'timestamp' в качестве индекса.

    Raises:
        KeyError: Если в датафрейме отсутствует столбец 'timestamp'.

    Example:
        >>> import pandas as pd
        >>> data = {'timestamp': ['2023-08-01 10:30:00', '2023-08-01 11:45:00']}
        >>> df = pd.DataFrame(data)
        >>> result = extract_datetime_components(df)
        >>> print(result)
    """
    # Преобразование 'timestamp' в datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Извлечение даты
    df['date'] = df['timestamp'].dt.date

    # Извлечение времени
    df['time'] = df['timestamp'].dt.time

    # Преобразование 'date' обратно в datetime для унификации типов
    df['date'] = pd.to_datetime(df['date'])

    # Установка 'timestamp' в качестве индекса
    df.set_index('timestamp', inplace=True)

    return df