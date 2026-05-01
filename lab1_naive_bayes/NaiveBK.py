
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Загружает данные из CSV, кодирует классы и преобразует признаки в float.
def load_and_prepare_data(filename):
    df = pd.read_csv(filename)
    mydata = df.values.tolist()
    mydata = encode_class(mydata)
    for i in range(len(mydata)):
        for j in range(len(mydata[i]) - 1):
            mydata[i][j] = float(mydata[i][j])
    return mydata
def encode_class(mydata):
    classes = []
    for i in range(len(mydata)):
        if mydata[i][-1] not in classes:
            classes.append(mydata[i][-1])
    for i in range(len(classes)):
        for j in range(len(mydata)):
            if mydata[j][-1] == classes[i]:
                mydata[j][-1] = i
    return mydata
# Разделяем данные на обучающую и тестовую выборки."
def splitting(mydata, ratio):
    train_num = int(len(mydata) * ratio)
    train = []
    test = list(mydata)
    while len(train) < train_num:
        index = random.randrange(len(test))
        train.append(test.pop(index))
    return train, test
#Группируем данные по классам
def groupUnderClass(mydata):
    data_dict = {}
    for i in range(len(mydata)):
        if mydata[i][-1] not in data_dict:
            data_dict[mydata[i][-1]] = []
        data_dict[mydata[i][-1]].append(mydata[i])
    return data_dict

def MeanAndStdDev(numbers):
    avg = np.mean(numbers)
    stddev = np.std(numbers)
    return avg, stddev

def MeanAndStdDevForClass(mydata):
    info = {}
    data_dict = groupUnderClass(mydata)
    for classValue, instances in data_dict.items():
        info[classValue] = [MeanAndStdDev(attribute) for attribute in zip(*[row[:-1] for row in instances])]
    return info
# Вычисляет значение функции Гаусса
def calculateGaussianProbability(x, mean, stdev):
    epsilon = 1e-10  # Added a small value to prevent division by zero
    expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev + epsilon, 2))))
    return (1 / (math.sqrt(2 * math.pi) * (stdev + epsilon))) * expo
#Вычисляет вероятность принадлежности к каждому классу для данного экземпляра
def calculateClassProbabilities(info, test):
    probabilities = {}
    for classValue, classSummaries in info.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, std_dev = classSummaries[i]
            x = test[i]
            probabilities[classValue] *= calculateGaussianProbability(x, mean, std_dev)
    return probabilities
#Предсказывает класс для данного экземпляра
def predict(info, test):
    probabilities = calculateClassProbabilities(info, test)
    bestLabel = max(probabilities, key=probabilities.get)
    return bestLabel
#Предсказывает класс для каждого экземпляра в тестовой выборке
def getPredictions(info, test):
    predictions = [predict(info, instance) for instance in test]  # Изменено здесь
    return predictions
#точность модели
def accuracy_rate(test, predictions):
    correct = sum(1 for i in range(len(test)) if test[i][-1] == predictions[i])
    return (correct / float(len(test))) * 100.0
#Функция визуализации решения
def plot_decision_boundaries(X_train, y_train, X_test, y_test, model, feature1_index=0, feature2_index=1, plot_symbols=True):
    """
    Визуализирует границы решений классификатора с использованием разных маркеров
    для разных классов.
    """

    # Объединяем обучающие и тестовые данные для определения границ графика
    X_combined = np.vstack((X_train, X_test))
    x_min, x_max = X_combined[:, feature1_index].min() - .5, X_combined[:, feature1_index].max() + .5
    y_min, y_max = X_combined[:, feature2_index].min() - .5, X_combined[:, feature2_index].max() + .5

    h = .02  # Шаг сетки
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = np.array([predict(model, np.array([x, y])) for x, y in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)

    # Цветовая карта
    cmap_light = ListedColormap(['#AAAAFF', '#AAFFAA', '#FFAAAA'])
    cmap_bold = ListedColormap(['#0000FF', '#00FF00', '#FF0000'])

    # границы решений
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light)

    # Маркеры для разных классов
    markers = ['o', 's', '^']  # Круг, квадрат, треугольник

    # Накладываем точки с разными маркерами
    if plot_symbols:
        class_labels = np.unique(y_test)
        for i, label in enumerate(class_labels):
            X_class = X_test[y_test == label]
            plt.scatter(X_class[:, feature1_index], X_class[:, feature2_index],
                        marker=markers[i % len(markers)],  # Используем разные маркеры
                        c=cmap_bold(i),  # Используем разные цвета
                        edgecolor='k',
                        s=30,
                        label=label)

        plt.legend(loc='upper right', title='Классы')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel('Sepal Length (Cm)')
    plt.ylabel('Sepal Width (Cm)')
    plt.title('График наивного байесовского классификатора')

    # Подсчет неверных классификаций
    predictions = getPredictions(model, X_test.tolist()) 
    predictions = np.array(predictions)
    incorrect_count = np.sum(predictions != y_test)
    incorrect_percentage = (incorrect_count / len(y_test)) * 100

    # Вывод информации об ошибках на график
    accuracy = accuracy_rate(test_data, predictions)
    text = f'Неверных классификаций: {incorrect_count} ({incorrect_percentage:.2f}%) \n Общее количество строк: {len(mydata)} \n Использвовано для обучения: {len(train_data)} \n Использовано для тестирования: {len(test_data)} \n Точность модели: {accuracy}'
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))

    plt.show()

# --- Основной код ---
filename = 'Iris-short.csv' 
mydata = load_and_prepare_data(filename)

# Преобразуем данные в numpy array (это нужно для plot_decision_boundaries)
mydata = np.array(mydata)

ratio = 0.4
train_data, test_data = splitting(mydata.tolist(), ratio)

X_train = np.array([row[:-1] for row in train_data])
y_train = np.array([row[-1] for row in train_data])
X_test = np.array([row[:-1] for row in test_data])
y_test = np.array([row[-1] for row in test_data])

print('Total number of examples:', len(mydata))
print('Training examples:', len(train_data))
print('Test examples:', len(test_data))

info = MeanAndStdDevForClass(train_data)
predictions = getPredictions(info, test_data)
accuracy = accuracy_rate(test_data, predictions)
print('Accuracy of the model:', accuracy)

# Визуализация решения
plot_decision_boundaries(X_train, y_train, X_test, y_test, info)
