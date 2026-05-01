# 3.	Найти дисперсию и стандартное отклонение любого параметра;
import pandas as pd
import numpy as np

if __name__ == '__main__':
  csv_file = 'output.csv'  # Путь к файлу CSV
  column_name = 'Height'  # Имя столбца
  df = pd.read_csv(csv_file)
  data = df[column_name].dropna().to_numpy()
  print('Дисперсия', np.var(data))
  print('Стандартное отклонение', np.std(data))