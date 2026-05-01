# import pandas as pd
# import seaborn as sns
# import numpy as np
# import matplotlib.pyplot as plt

# def линейная_зависимость(input_file, column_x, column_y):
#     """
#     Находит линейную зависимость между двумя параметрами и отображает график.

#     Args:
#         input_file (str): Путь к входному CSV файлу.
#         column_x (str): Название столбца для оси X.
#         column_y (str): Название столбца для оси Y.

#     Returns:
#         None
#     """
#     try:
#         df = pd.read_csv(input_file)
#     except FileNotFoundError:
#         print(f"Ошибка: Файл не найден: {input_file}")
#         return
#     except pd.errors.EmptyDataError:
#         print(f"Ошибка: Файл пуст: {input_file}")
#         return
#     except Exception as e:
#         print(f"Ошибка при чтении файла: {e}")
#         return

#     # Проверка наличия столбцов в DataFrame
#     if column_x not in df.columns or column_y not in df.columns:
#         print(f"Ошибка: Один или оба столбца '{column_x}' или '{column_y}' не найдены в файле.")
#         return

#     # Вычисление коэффициентов линейной зависимости
#     X = df[column_x]
#     y = df[column_y]
    
#     # Вычисление коэффициентов линейной регрессии
#     slope, intercept = np.polyfit(X, y, 1)

#     print(f"Коэффициент наклона (slope): {slope}")
#     print(f"Свободный член (intercept): {intercept}")

#     # Построение графика
#     plt.figure(figsize=(10, 6))
#     sns.regplot(x=column_x, y=column_y, data=df, marker='o', line_kws={"color": "red"})
#     plt.title(f'Линейная зависимость между {column_x} и {column_y}')
#     plt.xlabel(column_x)
#     plt.ylabel(column_y)
#     plt.grid()
#     plt.savefig(f'линейная_зависимость_{column_x}_{column_y}.png')
#     plt.show()

# # Пример использования
# линейная_зависимость('Iris.csv', 'SepalLengthCm', 'SepalWidthCm')  # Замените на нужные столбцы

from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Fish.csv")

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

print(df.head())
print(df['Species'].unique())

# Создаем и обучаем модель линейной регрессии
linreg = LinearRegression(fit_intercept = True) 

X2= df["Weight"]
y2= df["Height"]

x2_train, x2_test, y2_train, y2_test = train_test_split(X2, y2, 
        test_size = 0.3, random_state=42)

linreg.fit(np.array(x2_train).reshape(-1, 1), y2_train)
a = linreg.coef_[0]  # Коэффициент должен быть извлечен из массива
b = linreg.intercept_

# Вывод коэффициента наклона
print(f"Коэффициент наклона (slope): {a}")

# Визуализация: scatter plot данных и линия регрессии
plt.scatter(df["Weight"], df['Height'])

# Определяем диапазон значений для x, чтобы линия покрывала весь график
x_range = np.linspace(df["Weight"].min(), df["Weight"].max(), 100)
plt.plot(x_range, a * x_range + b, color="red", label=f'y = {a:.2f}x + {b:.2f}')  # Рисуем линию регрессии
plt.legend()

plt.xlabel("Weight")
plt.ylabel("Height")
plt.title("Linear Regression: Weight vs. Height")
plt.show()