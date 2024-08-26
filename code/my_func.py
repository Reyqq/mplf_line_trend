import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Dict, List, Tuple, Union, Callable, Optional
from tqdm.notebook import tqdm
import logging
pd.set_option('float_format', '{:.2f}'.format)

np.seterr(divide='ignore', invalid='ignore')



def extract_datetime_components(df: pd.DataFrame) -> pd.DataFrame:
      """
      
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


def set_plt_params(fontsize: float = 9.0, linewidth: float = 1.0, figsize: tuple = (12.8, 7.2)) -> None:
    """
    Устанавливает параметры для matplotlib.pyplot.

    Эта функция настраивает различные параметры для графиков matplotlib, включая
    цветовую схему, размеры шрифтов, стили линий и другие визуальные элементы.

    Args:
        fontsize (float, optional): Размер шрифта для меток осей и легенды. По умолчанию 9.0.
        linewidth (float, optional): Толщина линий на графиках. По умолчанию 1.0.
        figsize (tuple, optional): Размер фигуры (ширина, высота) в дюймах. По умолчанию (12.8, 7.2).

    Returns:
        None

    Example:
        >>> import matplotlib.pyplot as plt
        >>> set_plt_params(fontsize=10, linewidth=1.5, figsize=(10, 6))
        >>> plt.plot([1, 2, 3, 4])
        >>> plt.show()

    Note:
        Эта функция изменяет глобальные настройки matplotlib.pyplot.
        Все последующие графики будут использовать эти настройки,
        если они не будут изменены явно.
    """
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#dbdbdb", "#040404"])
    plt.rcParams['lines.linewidth'] = linewidth
    plt.rc("xtick", labelsize=fontsize, color="#2a2e39", labelcolor="#b2b5be")
    plt.rc("ytick", labelsize=fontsize, color="#2a2e39", labelcolor="#b2b5be")
    plt.rc("axes", facecolor="#181c27", edgecolor="#2a2e39", grid=True, titlesize=fontsize+2, labelcolor="#b2b5be")
    plt.rc("figure", figsize=figsize, facecolor="#181c27", edgecolor="#2a2e39")
    plt.rc("grid", color="#2a2e39", linestyle="dashed")
    plt.rc("legend", handleheight=1, handlelength=2, fontsize=fontsize)
    plt.rc("text", color="#b2b5be")