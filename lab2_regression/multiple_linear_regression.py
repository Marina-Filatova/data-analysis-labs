import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore") 
plt.style.use("fivethirtyeight")
df = pd.read_csv("output.csv")

le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

print(df.head())  # Показываем первые несколько строк датасета после преобразования
print(df['Species'].unique()) # Показываем уникальные значения в столбце Species после преобразования
# print(df.head())
# print(df.columns)
# print(df.describe().T)
# print(df.corr())

# plt.figure(figsize = (12,8))
# ax = sns.heatmap(df.drop(["Species"],axis=1).corr(), annot = True, fmt = ".2f")
# i, k = ax.get_ylim()
# ax.set_ylim(i+0.5, k-0.5)
# plt.show()
# print(np.linalg.det(df.corr()))
# df.boxplot(column="Species")
# df["Width"].hist(bins = 100)
# sns.displot(df["Species"])
# sns.pairplot(df.iloc[:,:])
# plt.show()

linreg = LinearRegression(fit_intercept = True) 
# Выбросим из набора целевой столбец, а также столбцы (Length1, Length3, сильно скореллированные между собой))
X = df.drop(["Species", "Length1", "Length3"], axis = 1)
y = df["Species"]
df.info()
X.shape, y.shape
# Сделаем стандартизацию (приведем данные к одному масштабу):
X_scal = StandardScaler().fit_transform(X)
# Разделим данные на тренировочные и тестовые (70% к 30%):
x_train, x_test, y_train, y_test = train_test_split(X_scal, y, 
        test_size = 0.2, random_state=42)
# Обучим модель линейной регрессии на тренировочных данных:
linreg.fit(x_train, y_train)
# Получаем предсказания для тестовой выборки
y_pred = linreg.predict(x_test)


# Визуализация: Фактические vs. Предсказанные значения
plt.figure(figsize=(10, 6))

# Создаем scatter plot: y_test - фактические значения, y_pred - предсказанные
plt.scatter(y_test, y_pred, alpha=0.7, label='Предсказания')


# Строим линию идеальных предсказаний (для сравнения)
plt.plot(np.linspace(y.min(), y.max(), 100), np.linspace(y.min(), y.max(), 100), color='red', label='Идеальные предсказания')

plt.xlabel("Фактические значения (Species)")
plt.ylabel("Предсказанные значения (Species)")
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
plt.title(f"Фактические vs. Предсказанные значения (RMSE: {rmse:.2f})")
plt.legend()  # Добавляем легенду
plt.grid(True)
plt.show()

print("среднеквадратичная ошибка", rmse)
print("коэффициент детерминации", r2_score(y_test, y_pred))
print(pd.DataFrame(linreg.coef_, X.columns, columns=["coef"]).sort_values(
    by="coef", ascending=False))