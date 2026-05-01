import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

test_data = pd.read_csv('thermal_imager-test.csv')
test_data = test_data.iloc[:, 8].values

plt.plot(test_data, color = 'black', label = "Real Data")

plt.title('Температура')
plt.show()
