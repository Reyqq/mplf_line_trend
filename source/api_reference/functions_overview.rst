.. _functions_overview:

=============
Обзор функций
=============


В этом разделе представлен краткий обзор основных функций, используемых в проекте.

- :func:`extract_datetime_components(df: pd.DataFrame) <my_func.extract_datetime_components>`: Преобразует столбец 'timestamp', извлекает компоненты даты и времени и устанавливает 'timestamp' в качестве индекса.
- :func:`set_plt_params(fontsize: float = 9.0, linewidth: float = 1.0, figsize: tuple = (12.8, 7.2)) <my_func.set_plt_params>`: Устанавливает параметры для matplotlib.pyplot.
- :func:`plot_candlestick_with_lines(df: pd.DataFrame, start_index: int = 0, end_index: Optional[int] = None, ...) <my_func.plot_candlestick_with_lines>`: Отображает свечной график с дополнительными линиями.
- :func:`calculate_relative_difference(open: float, close: float, low: float) <my_func.calculate_relative_difference>`: Вычисляет относительное изменение для актива на основе его цен открытия, закрытия и минимальной цены.
- :func:`linear_regression(df: pd.DataFrame, start_index: int = 0, ...) <my_func.linear_regression>`: Вычисляет значения линейной регрессии для заданного диапазона индексов DataFrame.
- :func:`linear_model(x: Union[float, np.ndarray], k: float, b: float) <my_func.linear_model>`: Вычисляет значение 'y' для линейной функции на основе входных параметров.
- :func:`get_coefs(x0: float, x1: float, y0: float, y1: float) <my_func.get_coefs>`: Вычисляет угловой коэффициент (k) и смещение (b) линейной функции.

Подробное описание каждой функции вы найдете в разделе ":ref:`api_reference/index:Справочник API`".
