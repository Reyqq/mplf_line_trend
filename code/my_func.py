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




def calculate_relative_difference(open: float, close: float, low: float) -> float:
    """
    Вычисляет относительное изменение для актива на основе его цен открытия, закрытия и минимальной цены.

    Эта функция рассчитывает относительное изменение цены актива в зависимости от того,
    закрылась ли цена выше или ниже цены открытия. Если цена закрытия выше цены открытия,
    используется формула (open - low) / open. В противном случае используется
    формула (close - low) / close.

    Args:
        open (float): Цена открытия актива.
        close (float): Цена закрытия актива.
        low (float): Минимальная цена актива за период.

    Returns:
        float: Относительное изменение цены актива.

    Examples:
        >>> calculate_relative_difference(100.0, 105.0, 98.0)
        0.02
        >>> calculate_relative_difference(100.0, 95.0, 93.0)
        0.021052631578947367

    Note:
        Эта функция предполагает, что все входные значения являются положительными числами.
        Отрицательные значения или значения, равные нулю, могут привести к неопределенному поведению.
    """
    if close > open:
        return (open - low) / open
    else:
        return (close - low) / close




def linear_regression(df: pd.DataFrame, start_index: int = 0, end_index: int = None, y0: float = 0, y1: float = 0) -> tuple:
    """
    Вычисляет значения линейной регрессии для заданного диапазона индексов DataFrame.

    Эта функция рассчитывает линейную регрессию на основе заданных начальной и конечной точек.
    Она вычисляет угловой коэффициент и свободный член линейной функции, а затем
    применяет эту функцию ко всем индексам в заданном диапазоне.

    Args:
        df (pd.DataFrame): DataFrame, содержащий данные для анализа.
        start_index (int, optional): Начальный индекс диапазона. По умолчанию 0.
        end_index (int, optional): Конечный индекс диапазона. Если None, используется последний индекс DataFrame.
        y0 (float, optional): Значение y в начальной точке. По умолчанию 0.
        y1 (float, optional): Значение y в конечной точке. По умолчанию 0.

    Returns:
        tuple: Кортеж, содержащий два элемента:
            - numpy.ndarray: Массив значений y, рассчитанных по линейной функции.
            - float: Угловой коэффициент k линейной функции.

    Raises:
        ValueError: Если end_index меньше start_index.
        IndexError: Если указанные индексы находятся за пределами DataFrame.

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': range(10)})
        >>> y, k = linear_regression(df, start_index=0, end_index=9, y0=1, y1=10)
        >>> print(f"Угловой коэффициент: {k}")
        Угловой коэффициент: 1.0
        >>> print(f"Значения y: {y}")
        Значения y: [ 1.  2.  3.  4.  5.  6.  7.  8.  9. 10.]

    Note:
        Эта функция предполагает, что индексы DataFrame являются числовыми и 
        последовательными. Если это не так, результаты могут быть неожиданными.
    """
    if end_index is None:
        end_index = df.index[-1]
    
    if end_index < start_index:
        raise ValueError("end_index должен быть больше или равен start_index")
    
    if start_index < df.index[0] or end_index > df.index[-1]:
        raise IndexError("Указанные индексы находятся за пределами DataFrame")

    x = df.loc[start_index:end_index].index.to_numpy()
    k = (y1 - y0) / (end_index - start_index)
    b = y0 - k * start_index
    y = k * x + b
    return y, k



def linear_model(x: Union[float, np.ndarray], k: float, b: float) -> Union[float, np.ndarray]:
    """
    Вычисляет значение 'y' для линейной функции на основе входных параметров.

    Эта функция реализует линейную модель вида y = kx + b, где k - угловой коэффициент,
    а b - свободный член. Функция может работать как с отдельными числами, так и с
    массивами numpy.

    Args:
        x (Union[float, np.ndarray]): Независимая переменная или массив независимых переменных.
        k (float): Угловой коэффициент линейной функции (наклон прямой).
        b (float): Свободный член линейной функции (точка пересечения с осью Y).

    Returns:
        Union[float, np.ndarray]: Значение(я) зависимой переменной y, вычисленное по формуле
        линейной функции. Тип возвращаемого значения соответствует типу входного параметра x.

    Examples:
        >>> linear_model(2, 3, 1)
        7
        >>> linear_model(np.array([1, 2, 3]), 2, 1)
        array([3, 5, 7])

    Note:
        Эта функция использует векторизованные операции numpy, поэтому она эффективна
        для больших массивов данных.
    """
    y = k * x + b
    return y




def get_coefs(x0: float, x1: float, y0: float, y1: float) -> Tuple[float, float]:
    """
    Вычисляет угловой коэффициент (k) и смещение (b) линейной функции.

    Эта функция рассчитывает параметры линейной функции вида y = kx + b
    на основе двух заданных точек (x0, y0) и (x1, y1). Функция использует
    формулы:
    k = (y1 - y0) / (x1 - x0)
    b = y0 - k * x0

    Args:
        x0 (float): Значение x первой точки.
        x1 (float): Значение x второй точки.
        y0 (float): Значение y первой точки.
        y1 (float): Значение y второй точки.

    Returns:
        Tuple[float, float]: Кортеж, содержащий два элемента:
            - float: Угловой коэффициент (k) линейной функции.
            - float: Смещение (b) линейной функции.

    Raises:
        ZeroDivisionError: Если x1 равно x0, что приводит к делению на ноль.

    Examples:
        >>> k, b = get_coefs(0, 2, 1, 5)
        >>> print(f"k = {k}, b = {b}")
        k = 2.0, b = 1.0

        >>> k, b = get_coefs(1, 3, 2, 8)
        >>> print(f"k = {k}, b = {b}")
        k = 3.0, b = -1.0

    Note:
        Эта функция предполагает, что входные точки различны по x (x0 != x1).
        В случае x0 = x1, будет вызвано исключение ZeroDivisionError.
    """
    if x1 == x0:
        raise ZeroDivisionError("x1 не может быть равно x0")

    k = (y1 - y0) / (x1 - x0)  # Вычисление углового коэффициента
    b = y0 - k * x0            # Вычисление смещения
    return k, b




def process_trendlines(trendline_dict: Dict[int, Dict[str, Tuple[int, int]]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Обрабатывает словарь с трендовыми линиями и извлекает пары координат для последующего использования.

    Функция анализирует словарь `trendline_dict`, находит ключи 'p2' и 'p3', и создаёт список кортежей с парами координат.
    Эти пары координат представляют собой начальную и конечную точки трендовых линий.

    Args:
        trendline_dict (Dict[int, Dict[str, Tuple[int, int]]]): Словарь, где ключами являются целые числа,
            а значениями - словари с ключами 'p1', 'p2', 'p3' и значениями в виде кортежей координат (x, y).

    Returns:
        List[Tuple[Tuple[int, int], Tuple[int, int]]]: Список кортежей, где каждый кортеж содержит две пары координат (x, y),
            соответствующие начальной и конечной точкам трендовых линий.

    Examples:
        >>> trendline_dict = {
        ...     0: {'p1': (0, 0), 'p2': (1, 1)},
        ...     1: {'p3': (2, 2)},
        ...     2: {'p1': (2, 2), 'p2': (3, 3)},
        ...     3: {'p3': (4, 4)}
        ... }
        >>> process_trendlines(trendline_dict)
        [((1, 1), (2, 2)), ((3, 3), (4, 4))]

    Note:
        - Функция предполагает, что ключи в словаре `trendline_dict` начинаются с 0 и идут последовательно.
        - Функция ищет 'p2' в предыдущем элементе и 'p3' в текущем элементе для формирования пары координат.
    """
    result_y = []
    value2 = None

    for item in range(1, len(trendline_dict)):
        if 'p2' in trendline_dict[item-1]:
            value2 = trendline_dict[item-1]['p2']

        if "p3" in trendline_dict[item]:
            value3 = trendline_dict[item]['p3']
            if value2 is not None:
                result_y.append((value2, value3))

    # Обработка последнего элемента
    if value2 is not None and "p3" in trendline_dict[len(trendline_dict) - 1]:
        last_value3 = trendline_dict[len(trendline_dict) - 1]['p3']
        result_y.append((value2, last_value3))

    return result_y





def check_trendline_touch(price: float, trendline_value: float, deviation: float) -> bool:
    """
    Проверяет, находится ли цена в пределах заданного процентного отклонения от значения линии тренда.

    Эта функция определяет, попадает ли заданная цена в диапазон, образованный значением линии тренда
    плюс-минус указанное процентное отклонение.

    Args:
        price (float): Цена закрытия свечи.
        trendline_value (float): Значение линии тренда в момент закрытия свечи.
        deviation (float): Допустимое отклонение от линии тренда в процентах (в десятичном формате).
                           Например, 0.05 для 5% отклонения.

    Returns:
        bool: True, если цена закрытия находится в пределах заданного отклонения от линии тренда,
              иначе False.

    Examples:
        >>> check_trendline_touch(100, 101, 0.02)
        True
        >>> check_trendline_touch(100, 105, 0.02)
        False

    Note:
        - Функция предполагает, что deviation передается в десятичном формате (например, 0.05 для 5%).
        - Функция симметрично проверяет отклонение как выше, так и ниже значения линии тренда.
    """
    lower_bound = trendline_value * (1 - deviation)
    upper_bound = trendline_value * (1 + deviation)
    return lower_bound <= price <= upper_bound



def check_point(i: int, close_data: List[float], open_data: List[float]) -> float:
    """
    Определяет минимальную цену за торговый день на основе индекса i.

    Эта функция сравнивает цены открытия и закрытия для заданного индекса
    и возвращает меньшую из них, что может быть использовано для анализа
    дневного ценового движения или расчета потенциальных убытков.

    Args:
        i (int): Индекс для доступа к элементам в массивах close_data и open_data.
        close_data (List[float]): Список цен закрытия.
        open_data (List[float]): Список цен открытия.

    Returns:
        float: Минимальная цена за день:
              - Цена открытия, если цена закрытия больше или равна цене открытия.
              - Цена закрытия, если цена закрытия меньше цены открытия.

    Raises:
        IndexError: Если индекс i выходит за пределы списков close_data или open_data.
        TypeError: Если i не является целым числом или если элементы close_data
                  или open_data не являются числами с плавающей точкой.

    Example:
        >>> close_prices = [101.5, 102.0, 100.5]
        >>> open_prices = [100.0, 102.5, 101.0]
        >>> check_point(0, close_prices, open_prices)
        100.0
        >>> check_point(1, close_prices, open_prices)
        102.0
        >>> check_point(2, close_prices, open_prices)
        100.5

    Note:
        - Функция предполагает, что списки close_data и open_data имеют одинаковую длину
          и содержат корректные данные.
        - Эта функция может быть полезна для анализа внутридневной волатильности
          или для определения потенциальных уровней поддержки.
    """
    
    if close_data[i] >= open_data[i]:
        return open_data[i]
    else:
        return close_data[i]



def process_dataframe(
    df: pd.DataFrame,
    start_index: int = 0,
    end_index: Optional[int] = None,
    skip_points: int = 0,
    touches: int = 1,
    deviation: float = 0,
    rel_diff: float = 0.00005
) -> pd.DataFrame:
    """
    Обрабатывает DataFrame с финансовыми данными и строит трендовые линии.

    Эта функция анализирует временной ряд финансовых данных, идентифицирует
    значимые точки тренда и строит линии тренда на основе заданных параметров.

    Args:
        df (pd.DataFrame): DataFrame с финансовыми данными. Ожидается наличие
                           колонок с ценами (например, 'Close', 'High', 'Low').
        start_index (int): Индекс первой обрабатываемой строки. По умолчанию 0.
        end_index (Optional[int]): Индекс последней обрабатываемой строки. 
                                   None означает обработку до конца DataFrame.
        skip_points (int): Количество точек для пропуска перед проверкой новой
                           линии тренда. Помогает избежать ложных сигналов.
        touches (int): Минимальное количество касаний цены линии тренда,
                       необходимое для подтверждения тренда.
        deviation (float): Допустимое абсолютное отклонение цены от линии тренда.
                           Используется для учета рыночного шума.
        rel_diff (float): Минимальная относительная разница цен для идентификации
                          точки тренда. Помогает отфильтровать незначительные колебания.

    Returns:
        pd.DataFrame: Обработанный DataFrame с добавленными колонками для
                      трендовых линий и их параметров.

    Raises:
        ValueError: Если входные параметры имеют недопустимые значения.

    Note:
        - Функция модифицирует входной DataFrame, добавляя новые колонки.
        - Рекомендуется предварительно очистить данные от выбросов и пропусков.
        - Для оптимальных результатов может потребоваться настройка параметров
          под конкретный финансовый инструмент и таймфрейм.
    """


    # 1. Проверка типов данных
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df должен быть pandas.DataFrame")

    # 2. Обработка пропущенных значений
    if df.isnull().values.any():
        logging.warning("В DataFrame обнаружены пропущенные значения")
        # df = df.dropna()   или df.fillna(method='ffill')

    # 3. Проверка диапазона индексов
    if start_index < 0 or (end_index is not None and end_index > len(df)):
        raise ValueError("Неверно указаны индексы start_index или end_index")

    # 4. Валидация параметров
    if skip_points < 0 or touches < 1 or deviation < 0 or rel_diff <= 0:
        raise ValueError("Один из параметров имеет недопустимое значение")


    if end_index is None:
        end_index = len(df)

    trend_lines = [np.nan] * len(df.iloc[start_index:end_index])


    result = []
    current_points = []
    historical_points = []
    trendline_dict = {}


    skp_points = 0
    touch_count = 0

    open_data = df['open']
    close_data = df['close']
    low_data = df['low']
    high_data = df['high']


    for i in tqdm(range(start_index, end_index)):



        if len(current_points) == 0:
            result.append(np.nan)
            relative_difference = calculate_relative_difference(open_data[i], close_data[i], low_data[i])
            if relative_difference >= rel_diff:
                historical_points.append((i, low_data[i]))
                current_points.append((i, low_data[i]))



        elif len(current_points) == 2:
          x_curr = np.arange(current_points[0][0], i + 1)
          y_curr = linear_model(x_curr, k=k, b=b)


          if skp_points > skip_points:
            relative_difference = calculate_relative_difference(open_data[i], close_data[i], low_data[i])
            if relative_difference >= rel_diff:
                historical_points.append((i, low_data[i]))
                current_points.append((i, low_data[i]))
                del current_points[0]


                k1, b1 = get_coefs(current_points[0][0], current_points[1][0], current_points[0][1], current_points[1][1])
                x = np.arange(current_points[0][0], current_points[1][0] + 1)
                y = linear_model(x, k=k1, b=b1)
                if k1 >= 0:

                    for j in range(current_points[0][0], current_points[1][0] + 1):

                        # if check_trendline_touch(y[j - current_points[0][0]], low_data[j], deviation):
                        #     touch_count += 1

                        if y[j - current_points[0][0]] > low_data[j]:
                            touch_count += 1

                        if touch_count >= touches :
                            del current_points[0]
                            result.append(y_curr[-1])
                            trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
                            skp_points = 0
                            touch_count = 0
                            break

                    else:
                      for z in range(current_points[0][0], current_points[1][0] + 1):
                        trend_lines[z - start_index] = round(y[z - current_points[0][0]], 2)
                      skp_points = 0
                      touch_count = 0
                      k, b = get_coefs(current_points[0][0], current_points[1][0], current_points[0][1], current_points[1][1])
                      trendline_dict[len(trendline_dict)] = {'p1': (str(df.index[current_points[0][0]]), round(y[0], 2)), 'p2': (str(df.index[current_points[1][0]]), round(y[-1], 2))}
                      result.append((i, low_data[i]))
                      continue
                elif check_point(i) > y_curr[-1]:
                    result.append(y_curr[-1])
                    trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
                    del current_points[0]
                    skp_points = 0
                    continue
                else:
                  result.append(np.nan)
                  del current_points[0]
                  skp_points = 0
                  continue

            elif check_point(i) > y_curr[-1]:
              result.append(y_curr[-1])
              trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
              continue
            else:
              skp_points = 0
              del current_points[0]
              result.append(np.nan)
              continue
          elif check_point(i) > y_curr[-1]:
              skp_points += 1
              result.append(y_curr[-1])
              trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
              skp_points += 1

          else:
            result.append(np.nan)
            skp_points = 0
            del current_points[0]




        elif len(current_points) == 1 and result[-1] is not np.nan and len(historical_points) >= 2:
          x_curr = np.arange(current_points[0][0], i + 1)
          y_curr = linear_model(x_curr, k=k, b=b)

          if skp_points > skip_points:
            relative_difference = calculate_relative_difference(open_data[i], close_data[i], low_data[i])
            if relative_difference >= rel_diff:
                historical_points.append((i, low_data[i]))
                current_points.append((i, low_data[i]))

                k1, b1 = get_coefs(current_points[0][0], current_points[1][0], current_points[0][1], current_points[1][1])
                x = np.arange(current_points[0][0], current_points[1][0] + 1)
                y = linear_model(x, k=k1, b=b1)
                if k1 >= 0:


                    for j in range(current_points[0][0], current_points[1][0] + 1):

                        # if check_trendline_touch(y[j - current_points[0][0]], low_data[j], deviation):
                        #     touch_count += 1

                        if y[j - current_points[0][0]] > low_data[j]:
                            touch_count += 1

                        if touch_count >= touches:
                            result.append(y_curr[-1])
                            trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
                            skp_points = 0
                            del current_points[0]
                            touch_count = 0
                            break

                    else:
                      for z in range(current_points[0][0], current_points[1][0] + 1):
                        trend_lines[z - start_index] = round(y[z - current_points[0][0]], 2)
                      skp_points = 0
                      touch_count = 0
                      k, b = get_coefs(current_points[0][0], current_points[1][0], current_points[0][1], current_points[1][1])
                      trendline_dict[len(trendline_dict)] = {'p1': (str(df.index[current_points[0][0]]), round(y[0], 2)), 'p2': (str(df.index[current_points[1][0]]), round(y[-1], 2))}
                      result.append((i, low_data[i]))

                elif check_point(i) > y_curr[-1]:
                  result.append(y_curr[-1])
                  trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
                  del current_points[0]
                  skp_points = 0

                else:
                  del current_points[0]
                  skp_points = 0
                  result.append(np.nan)


            elif check_point(i) > y_curr[-1]:
              result.append(y_curr[-1])
              trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
              continue

            else:
              result.append(np.nan)



          elif check_point(i) > y_curr[-1]:
            result.append(y_curr[-1])
            trendline_dict[len(trendline_dict)] = {'p3': (str(df.index[i]), round(y_curr[-1], 2))}
            skp_points += 1
            continue

          else:
            result.append(np.nan)
            skp_points += 1
            continue





        elif len(current_points) == 1:
          if skp_points > skip_points:
            relative_difference = calculate_relative_difference(open_data[i], close_data[i], low_data[i])
            if relative_difference >= rel_diff:
                historical_points.append((i, low_data[i]))
                current_points.append((i, low_data[i]))

                k1, b1 = get_coefs(current_points[0][0], current_points[1][0], current_points[0][1], current_points[1][1])
                x = np.arange(current_points[0][0], current_points[1][0] + 1)
                y = linear_model(x, k=k1, b=b1)
                if k1 >= 0:


                    for j in range(current_points[0][0], current_points[1][0] + 1):

                        # if check_trendline_touch(y[j - current_points[0][0]], low_data[j], deviation):
                        #     touch_count += 1

                        if y[j - current_points[0][0]] > low_data[j]:
                            touch_count += 1

                        if touch_count >= touches:
                            result.append(np.nan)
                            skp_points = 0
                            del current_points[0]
                            touch_count = 0
                            break

                    else:
                      for z in range(current_points[0][0], current_points[1][0] + 1):
                        trend_lines[z - start_index] = round(y[z - current_points[0][0]], 2)
                      skp_points = 0
                      touch_count = 0
                      k, b = get_coefs(current_points[0][0], current_points[1][0], current_points[0][1], current_points[1][1])
                      trendline_dict[len(trendline_dict)] = {'p1': (str(df.index[current_points[0][0]]), round(y[0], 2)), 'p2': (str(df.index[current_points[1][0]]), round(y[-1], 2))}
                      result.append((i, low_data[i]))

                else:
                  del current_points[0]
                  skp_points = 0
                  result.append(np.nan)


            else:
              result.append(np.nan)



          else:
              result.append(np.nan)
              skp_points += 1



    plot_candlestick_with_lines_final = plot_candlestick_with_lines(df=df, start_index=start_index, end_index=end_index, line_data=trend_lines,trendline_dict=trendline_dict)

    return plot_candlestick_with_lines_final









    