# 2.	Произвести предобработку данных (удалить шумы, дубликаты);
import pandas as pd

def очистить_csv(input_file, output_file, columns_to_check=None, remove_empty_rows=True):
    """
    Удаляет повторы и шумы и сохраняет результат в новый файл.

    Args:
        input_file (str): Путь к входному CSV файлу.
        output_file (str): Путь к выходному CSV файлу.
        columns_to_check (list, optional): Список названий столбцов, по которым проверять дубликаты.
                                           Если None, проверяются все столбцы. Defaults to None.
        remove_empty_rows (bool, optional):  Удалять ли пустые строки (все значения NaN). Defaults to True.

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

    print(f"Исходное количество строк: {len(df)}")

    # Удаление пустых строк
    if remove_empty_rows:
        df.dropna(how='all', inplace=True)  # Удаляем строки, где все значения NaN
        print(f"Количество строк после удаления пустых строк: {len(df)}")

    # Удаление дубликатов
    if columns_to_check:
        df.drop_duplicates(subset=columns_to_check, inplace=True)
        print(f"Количество строк после удаления дубликатов по столбцам {columns_to_check}: {len(df)}")
    else:
        df.drop_duplicates(inplace=True)
        print(f"Количество строк после удаления всех дубликатов: {len(df)}")

    # Удаление выбросов
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        print(f"Количество строк после удаления выбросов в столбце {column}: {len(df)}")

    try:
        df.to_csv(output_file, index=False)
        print(f"Файл успешно очищен и сохранен в: {output_file}")
    except Exception as e:
        print(f"Ошибка при записи в файл: {e}")

очистить_csv('Fish.csv', 'output.csv')  # Удаление всех дубликатов, пустых строк и выбросов