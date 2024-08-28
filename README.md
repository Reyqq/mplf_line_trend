# mplfinance_line_trend

![Logo](docs/source/image/grafic.png)

**mplf_line_trend** - это Python библиотека для анализа и визуализации трендов финансовых данных с использованием mplfinance.

## Особенности

- Анализ трендов на основе исторических данных о ценах
- Визуализация трендов с помощью mplfinance
- Настраиваемые параметры для точной настройки анализа
- Поддержка различных форматов входных данных
- Простой и интуитивно понятный API

## Установка

Установите mplf_line_trend с помощью pip:

.. code-block:: python
   pip install mplf-line-trend


## Документация

Полная документация доступна на: https://reyqq.github.io/mplf_line_trend/

## Быстрый старт

Вот простой пример использования mplf_line_trend:

.. code-block:: python

   import pandas as pd
   from mplf_line_trend import process_dataframe, extract_datetime_components, set_plt_params

   # Загрузка данных
   df = pd.read_parquet('path/to/your/data.parquet')

   # Подготовка данных
   df = extract_datetime_components(df)
   set_plt_params()

   # Анализ и визуализация трендов
   processed_df = process_dataframe(
        df=df,
        start_index=0,
        end_index=300,
        skip_points=2,
        touches=1,
        deviation=0.000000000003,
        rel_diff=0.00005
    )


## Вклад в проект

Мы приветствуем вклад в развитие проекта! Пожалуйста, ознакомьтесь с нашим руководством по внесению вклада для получения дополнительной информации.


## Контакты

Rey - lorkaaqq@gmail.com 

GBM|e^i*pi = -1 - 

Ссылка на проект: https://github.com/Reyqq/mplf_line_trend

