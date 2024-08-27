===================
mplf_line_trend API
===================

Этот раздел содержит подробную документацию по всем вспомогательным функциям, доступным в нашей библиотеке.

============
Документация
============

Этот раздел содержит подробное описание всех функций, реализованных в проекте.

- :func:`extract_datetime_components <my_func.extract_datetime_components>`: Преобразует столбец 'timestamp', извлекает компоненты даты и времени и устанавливает 'timestamp' в качестве индекса.
- :func:`set_plt_params <my_func.set_plt_params>`: Устанавливает параметры для matplotlib.pyplot.
- :func:`plot_candlestick_with_lines <my_func.plot_candlestick_with_lines>`: Отображает свечной график с дополнительными линиями.
- :func:`calculate_relative_difference <my_func.calculate_relative_difference>`: Вычисляет относительное изменение для актива на основе его цен открытия, закрытия и минимальной цены.
- :func:`linear_regression <my_func.linear_regression>`: Вычисляет значения линейной регрессии для заданного диапазона индексов DataFrame.
- :func:`linear_model <my_func.linear_model>`: Вычисляет значение 'y' для линейной функции на основе входных параметров.
- :func:`get_coefs <my_func.get_coefs>`: Вычисляет угловой коэффициент (k) и смещение (b) линейной функции.
- :func:`process_trendlines <my_func.process_trendlines>`: Обрабатывает словарь с трендовыми линиями и извлекает пары координат для последующего использования.
- :func:`check_trendline_touch <my_func.check_trendline_touch>`: Проверяет, находится ли цена в пределах заданного процентного отклонения от значения линии тренда.


.. automodule:: my_func
   :members:
   :undoc-members:
   :show-inheritance:
