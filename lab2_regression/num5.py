# 6. Построить гистограмму, определить тип распределения одного из параметров;
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

def analys(input_file, column):
    """
    Строит гистограмму и определяет тип распределения заданного параметра.

    Args:
        input_file (str): Путь к входному CSV файлу.
        column (str): Название столбца для анализа.

    Returns:
        None
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден: {input_file}")
        return
    except pd.errors.EmptyDataError:
        print(f"Ошибка: Файл пуст: {input_file}")
        return
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        return

    # Проверка наличия столбца в DataFrame
    if column not in df.columns:
        print(f"Ошибка: Столбец '{column}' не найден в файле.")
        return

    # Построение гистограммы
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=30, kde=True)
    plt.title(f'Гистограмма распределения {column}')
    plt.xlabel(column)
    plt.ylabel('Частота')
    plt.grid()
    plt.savefig(f'гистограмма_{column}.png')
    plt.show()

    # Тест Шапиро-Уилка для проверки нормальности
    stat, p_value = stats.shapiro(df[column].dropna())
    alpha = 0.05
    if p_value > alpha:
        print(f"Распределение {column} нормально (p-value = {p_value:.5f})")
    else:
        print(f"Распределение {column} ненормально (p-value = {p_value:.5f})")

analys('output.csv', 'Length2')  




