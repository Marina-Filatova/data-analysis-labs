import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os

# импорт данных для обучения
training_data = pd.read_csv('thermal_imager-train.csv')
training_data = training_data.iloc[:, 7].values #8й столбец, wind_speed
# print(type(training_data))

# нормализация данных
scaler = MinMaxScaler() 
training_data = scaler.fit_transform(training_data.reshape(-1, 1)) #каждое наблюдение в диапазоне от -1 до 1

# Указание количества временных шагов
x_training_data = [] #40 предыдущих измерений. Это данные, которые рекуррентная нейронная сеть будет использовать для прогнозирования.
y_training_data =[] #Это точка данных, которую пытается предсказать рекуррентная нейронная сеть.

for i in range(40, len(training_data)):
    x_training_data.append(training_data[i-40:i, 0])
    y_training_data.append(training_data[i, 0])

x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)
# print(x_training_data.shape)
# print(y_training_data.shape)
x_training_data = np.reshape(x_training_data, (x_training_data.shape[0], x_training_data.shape[1], 1))
# print(x_training_data.shape)
rnn = Sequential()
#Первый слой LSTM
rnn.add(LSTM(units = 30, return_sequences = True, input_shape = (x_training_data.shape[1], 1)))
rnn.add(Dropout(0.2))
#Еще 3 слоя LSTM с регуляризацией выпадения
for i in [True, True, False]:
    rnn.add(LSTM(units = 30, return_sequences = i))
    rnn.add(Dropout(0.2))
# выходной слой
rnn.add(Dense(units = 1))

rnn.compile(optimizer = 'adam', loss = 'mean_squared_error')
# обучение
# rnn.fit(x_training_data, y_training_data, epochs = 100, batch_size = 64)
early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
history = rnn.fit(x_training_data, y_training_data, epochs=100, batch_size=32, callbacks=[early_stopping])

# импорт данных для тестирования
test_data = pd.read_csv('thermal_imager-test.csv')
test_data = test_data.iloc[:, 7].values
# plt.plot(test_data)
# plt.show()
unscaled_x_training_data = pd.read_csv('thermal_imager-train.csv')
unscaled_test_data = pd.read_csv('thermal_imager-test.csv')
all_data = pd.concat((unscaled_x_training_data.iloc[:, 7], unscaled_test_data.iloc[:, 7]), axis = 0)
print(all_data.shape)
print(all_data)
x_test_data = all_data[len(all_data) - len(test_data) - 40:].values
x_test_data = np.reshape(x_test_data, (-1, 1))
# масштабирование данных для тестирования
x_test_data = scaler.transform(x_test_data)
final_x_test_data = []

for i in range(40, len(x_test_data)):
    final_x_test_data.append(x_test_data[i-40:i, 0])

final_x_test_data = np.array(final_x_test_data)
final_x_test_data = np.reshape(final_x_test_data, (final_x_test_data.shape[0], final_x_test_data.shape[1], 1))

predictions = rnn.predict(final_x_test_data)    
unscaled_predictions = scaler.inverse_transform(predictions)
plt.plot(unscaled_predictions)

plt.plot(unscaled_predictions, color = '#135485', label = "Predictions")
# plt.plot(test_data, color = 'black', label = "Real Data")

plt.title('Wind speed')

# Построение графика loss
plt.figure(figsize=(10, 6))  # Создаем новую фигуру для графика loss
plt.plot(history.history['loss'], label='Loss')  # Используем history.history['loss']
plt.title('График изменения Loss')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()
plt.show()
