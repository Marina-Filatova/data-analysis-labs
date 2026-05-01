#Найти среднее значение, медиану, экстремумы любого параметра;
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks


def extrema(csv_file, column_name, prominence=None, distance=None):
  try:
    df = pd.read_csv(csv_file)
  except FileNotFoundError:
    print(f"Ошибка: Файл '{csv_file}' не найден.")
    return [], []
  except pd.errors.EmptyDataError:
    print(f"Ошибка: Файл '{csv_file}' пуст.")
    return [], []
  except pd.errors.ParserError:
    print(f"Ошибка: Не удалось прочитать файл '{csv_file}'. Убедитесь, что это корректный CSV файл.")
    return [], []
  except Exception as e:
    print(f"Произошла ошибка при чтении файла: {e}")
    return [], []

  if column_name not in df.columns:
    print(f"Ошибка: Столбец '{column_name}' не найден в CSV файле.")
    return [], []

  data = df[column_name].dropna().to_numpy() # Извлекаем данные из столбца и удаляем NaN

  # Находим максимумы
  peaks, _ = find_peaks(data, prominence=prominence, distance=distance)

  # Находим минимумы 
  valleys, _ = find_peaks(-data, prominence=prominence, distance=distance)

  return peaks, valleys

if __name__ == '__main__':
  csv_file = 'Fish.csv'  # Путь к файлу CSV
  column_name = 'Height'  # Имя столбца
  prominence = 10  # Значение аргумента prominence для нахождения экстремумов
  distance = 20  # Значение аргумента distance для нахождения экстремумов

  peaks, valleys = extrema(csv_file, column_name, prominence=prominence, distance=distance)

  try:
    df = pd.read_csv(csv_file)
    data = df[column_name].dropna().to_numpy()
    mean = np.mean(data)
    print("Среднее значение:", mean)

    median = np.median(data)
    print("Медиана:", median)

    print("\nЗначения максимумов:")
    for i in peaks:
        print(f"Индекс: {i}, Значение: {data[i]}")

    print("\nЗначения минимумов:")
    for i in valleys:
        print(f"Индекс: {i}, Значение: {data[i]}")

  except Exception as e:
      print(f"Произошла ошибка при выводе значений: {e}")
  
 
    
  
