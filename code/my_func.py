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







def plot_candlestick_with_lines(
    df: pd.DataFrame,
    start_index: int = 0,
    end_index: Optional[int] = None,
    chart_title: Optional[str] = None,
    ylabel: Optional[str] = None,
    xlabel: Optional[str] = None,
    title_fontsize: int = 18,
    label_color: str = "#b2b5be",
    line_data: Optional[pd.Series] = None,
    line_processor: Optional[Callable[[dict], list]] = None,
    trendline_dict: Optional[dict] = None
) -> None:
    """
    Отображает свечной график с дополнительными линиями.

    Эта функция создает свечной график на основе предоставленных данных и добавляет
    дополнительные линии тренда и другие визуальные элементы.

    Args:
        df (pd.DataFrame): Датафрейм с данными для графика. Должен содержать столбцы
            'Open', 'High', 'Low', 'Close' для построения свечей.
        start_index (int, optional): Начальный индекс данных в датафрейме для отображения
            на графике. По умолчанию 0.
        end_index (int, optional): Конечный индекс данных в датафрейме для отображения
            на графике. По умолчанию None (до конца датафрейма).
        chart_title (str, optional): Заголовок графика. По умолчанию None.
        ylabel (str, optional): Название оси Y. По умолчанию None.
        xlabel (str, optional): Название оси X. По умолчанию None.
        title_fontsize (int, optional): Размер шрифта заголовка. По умолчанию 18.
        label_color (str, optional): Цвет текста названий осей и заголовка.
            По умолчанию "#b2b5be".
        line_data (pd.Series, optional): Данные для отображения дополнительной линии
            на графике. По умолчанию None.
        line_processor (Callable[[dict], list], optional): Функция для обработки линий.
            По умолчанию None.
        trendline_dict (dict, optional): Словарь с данными для линий, который будет
            обработан функцией line_processor. По умолчанию None.

    Raises:
        ValueError: Если trendline_dict не определен или не найдено подходящих линий.

    Returns:
        None: Функция отображает график, но не возвращает значение.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'Open': [10, 11, 12],
        ...     'High': [12, 13, 14],
        ...     'Low': [9, 10, 11],
        ...     'Close': [11, 12, 13]
        ... })
        >>> plot_candlestick_with_lines(df, chart_title="Пример графика")

    Note:
        Эта функция использует библиотеки matplotlib и mplfinance для создания графика.
        Убедитесь, что эти библиотеки установлены перед использованием функции.
    """
    if trendline_dict is None:
        raise ValueError("trendline_dict не определён. Пожалуйста, предоставьте словарь с данными для линий.")

    if line_processor is None:
        line_processor = process_trendlines(trendline_dict)

    if len(line_processor) < 1:
       raise ValueError("Не найдено подходящих линий.")

    market_colors = mpf.make_marketcolors(up="#ffffff", down="#1976d2", edge="#2a2e39", wick="#787b86")

    style = {
        "xtick.labelcolor": '#b2b5be',
        "ytick.labelcolor": '#b2b5be',
        "xtick.color": '#2a2e39',
        "ytick.color": '#2a2e39',
    }
    chart_style = mpf.make_mpf_style(facecolor="#181c27", figcolor="#181c27", edgecolor="#2a2e39", 
                                     gridcolor="#2a2e39", gridstyle="dashed", rc=style, 
                                     marketcolors=market_colors, y_on_right=True)

    addplot = mpf.make_addplot(line_data, type='line', color='r', width=0.5)

    fig, axlist = mpf.plot(df.iloc[start_index:end_index],
                           type="candle", 
                           alines=dict(alines=line_processor, colors='g', linewidths=0.5),
                           ylabel="Price", 
                           show_nontrading=True, 
                           figratio=(25.6, 14.4), 
                           figscale=2, 
                           style=chart_style, 
                           returnfig=True, 
                           scale_padding={"top": 1.5}, 
                           tight_layout=True, 
                           addplot=addplot, 
                           xrotation=0)

    axlist[0].set_ylabel(ylabel if ylabel is not None else None, color=label_color)
    axlist[0].set_xlabel(xlabel if xlabel is not None else None, color=label_color)

    if chart_title:
        fig.suptitle(chart_title, color=label_color, fontsize=title_fontsize)

    plt.show()









    