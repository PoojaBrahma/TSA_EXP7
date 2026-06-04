# Ex.No: 07 AUTO REGRESSIVE MODEL
### Date: 04.06.2026

### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import mean_squared_error

data = pd.read_csv("Google_Stock_Price_Test.csv")
data['Date'] = pd.to_datetime(data['Date'], format='mixed')
data = data.sort_values('Date')
data.set_index('Date', inplace=True)
data = data.asfreq('B')
data = data.ffill()
data = data[['Close']]
data['Close'] = np.log(data['Close'])
data['Close'] = data['Close'].diff()
data = data.dropna()
print(data.head())
result = adfuller(data['Close'])

print('ADF Statistic:', result[0])
print('p-value:', result[1])
x = int(0.8 * len(data))

train_data = data.iloc[:x]
test_data = data.iloc[x:]
lag_order = 5

model = AutoReg(
    train_data['Close'],
    lags=lag_order
)

model_fit = model.fit()
plt.figure(figsize=(10, 6))

plot_acf(
    data['Close'],
    lags=min(10, len(data)-1),
    alpha=0.05
)

plt.title('Autocorrelation Function (ACF)')

plt.show()

plt.figure(figsize=(10, 6))

plot_pacf(
    data['Close'],
    lags=min(10, len(data)//2 - 1),
    alpha=0.05
)

plt.title('Partial Autocorrelation Function (PACF)')

plt.show()

predictions = model_fit.predict(
    start=len(train_data),
    end=len(train_data)+len(test_data)-1,
    dynamic=False
)

mse = mean_squared_error(
    test_data['Close'],
    predictions
)

print('Mean Squared Error (MSE):', mse)

plt.figure(figsize=(12, 6))

plt.plot(
    test_data.index,
    test_data['Close'],
    label='Test Data - Google Stock'
)

plt.plot(
    test_data.index,
    predictions,
    label='Predictions - Google Stock',
    linestyle='--'
)

plt.xlabel('Date')

plt.ylabel('Log Differenced Close Price')

plt.title('AR Model Predictions vs Test Data - Google Stock')

plt.legend()

plt.grid()

plt.show()
```
### OUTPUT:

GIVEN DATA

<img width="433" height="165" alt="image" src="https://github.com/user-attachments/assets/001aa802-bbe2-44b3-9423-3da121af8ec6" />

PACF - ACF

<img width="619" height="565" alt="Screenshot 2026-05-29 214814" src="https://github.com/user-attachments/assets/4f372992-6c90-4b76-9fbd-47a816867905" />

PREDICTION

<img width="648" height="346" alt="image" src="https://github.com/user-attachments/assets/a2325ddf-7516-4cd3-bef8-264d5ffe64e0" />

Performance Measure (MSE)

<img width="468" height="23" alt="image" src="https://github.com/user-attachments/assets/b5ecae41-77bc-44ba-9e9f-343d51817880" />

### RESULT:
Thus we have successfully implemented the auto regression function using python.
