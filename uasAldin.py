import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstest, norm
from sklearn.linear_model import LinearRegression

dataWalmart = pd.read_csv('Walmart.csv')

#No1
#Poin a
# -	Weekly_Sales
# -	Customer Price Index
# -	Temperature
# -	Fuel_Price
# -	Unemployment
# -	Holiday_Flag


#Poin b
dataStore4 = dataWalmart[dataWalmart['Store'] == 4]

#Poin 1B
weekly_sales_stats = dataStore4['Weekly_Sales'].describe()
# print('Weekly Sales Stats : ', weekly_sales_stats)
cpi_stats = dataStore4['CPI'].describe()
# print('CPI Stats : ', cpi_stats)
temperature_stats = dataStore4['Temperature'].describe()
# print('Temp Stats', temperature_stats)
holiday_flags_stats = dataStore4['Holiday_Flag'].describe()
# print('holiday flags stats : ', holiday_flags_stats)
fuel_price_stats = dataStore4['Fuel_Price'].describe()
# print('fuel price stats : ', fuel_price_stats)
unemployment_stats = dataStore4['Unemployment'].describe()
# print('unemployment stats : ', unemployment_stats)


#1C
#IQR Fuel_Price
q1Fuel = dataStore4['Fuel_Price'].quantile(0.25)
q2Fuel = dataStore4['Fuel_Price'].quantile(0.50)
q3Fuel = dataStore4['Fuel_Price'].quantile(0.75)
iqrFuel = q3Fuel - q1Fuel
# print('q1 Fuel_Price : ', q1Fuel)
# print('q2 Fuel_Price : ', q2Fuel)
# print('q3 Fuel_Price : ', q3Fuel)
# print('IQR Untuk Fuel_Price : ', iqrFuel)

#IQR untuk CPI
q1Cpi = dataStore4['CPI'].quantile(0.25)
q2Cpi = dataStore4['CPI'].quantile(0.50)
q3Cpi = dataStore4['CPI'].quantile(0.75)
iqrCpi = q3Cpi - q1Cpi
# print('q1 Cpi : ', q1Cpi)
# print('q2 Cpi : ', q2Cpi)
# print('q3 Cpi : ', q3Cpi)
# print('IQR Untuk CPI : ', iqrCpi)

#IQR untuk Unemployment
q1Unemploy = dataStore4['Unemployment'].quantile(0.25)
q2Unemploy = dataStore4['Unemployment'].quantile(0.50)
q3Unemploy = dataStore4['Unemployment'].quantile(0.75)
iqrUnemploy = q3Unemploy - q1Unemploy
# print('q1 Unemployment : ', q1Unemploy)
# print('q2 Unemployment : ', q1Unemploy)
# print('q3 Unemployment : ', q1Unemploy)
# print('IQR Untuk Unemployment : ', iqrUnemploy)

#1D Varians
holiday_variance = dataWalmart.groupby('Holiday_Flag')['Weekly_Sales'].var()
# print("Variance Description:")
# for flag, variance in holiday_variance.items():
#     if flag == 1:
#         print("Holiday Week:")
#     else:
#         print("Non-Holiday Week:")
#     print("Variance:", variance)

#1e
stores_mean = dataWalmart.groupby('Store')['Weekly_Sales'].mean()
is_stores_mean_equal = stores_mean.nunique() == 1
# print(stores_mean)
# if is_stores_mean_equal:
#     print("Rata-rata semua store sama")
# else:
#     print("Rata-Rata semua toko tidak sama")

#1f
max_cpi_by_store = dataWalmart.groupby('Store')['CPI'].max()
higher_cpi_by_store = max_cpi_by_store.idxmax()
# print('Store : ', higher_cpi_by_store)
higher_cpi_value = max_cpi_by_store.max()
# print('CPI : ', higher_cpi_value)

#1g
mean_cpi_holiday = dataWalmart[dataWalmart['Holiday_Flag'] == 1]['CPI'].mean()
mean_cpi_non_holiday = dataWalmart[dataWalmart['Holiday_Flag'] == 0]['CPI'].mean()
# if mean_cpi_holiday > mean_cpi_non_holiday:
#     print("Rata-rata CPI pada holiday week lebih tinggi.")
# elif mean_cpi_holiday < mean_cpi_non_holiday:
#     print("Rata-rata CPI pada non-holiday week lebih tinggi.")
# else:
#     print("Rata-rata CPI pada holiday week dan non-holiday week sama.")

#2
weekly_sales = dataWalmart['Weekly_Sales']
fuel_price = dataWalmart['Fuel_Price']
alpha = 0.05

statistic, p_value = kstest(weekly_sales, norm.fit(weekly_sales))
print("Uji Normalitas Weekly Sales:")
print(f"Statistic: {statistic}")
print(f"P-value: {p_value}")
if p_value > alpha:
    print("Weekly Sales didistribusikan secara normal")
else:
    print("Weekly Sales tidak didistribusikan secara normal")

statistic, p_value = kstest(fuel_price, norm.fit(fuel_price))
print("Uji Normalitas Fuel Price:")
print(f"Statistic: {statistic}")
print(f"P-value: {p_value}")
if p_value > alpha:
    print("Fuel Price didistribusikan secara normal")
else:
    print("Fuel Price tidak didistribusikan secara normal")


#3
#3a
correlation = dataWalmart[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']].corr()
print("Nilai korelasi antara variabel independen dan variabel dependen:")
print(correlation['Weekly_Sales'])

#3b
correlation = dataWalmart[['Holiday_Flag', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Weekly_Sales']].corr()
negative_correlations = correlation[correlation['Weekly_Sales'] < 0]
negative_correlations = negative_correlations['Weekly_Sales'].drop('Weekly_Sales', errors='ignore')
if negative_correlations.empty:
    print("Tidak ada pasangan variabel independen dan dependen dengan korelasi negatif.")
else:
    print("Pasangan variabel independen dan dependen dengan korelasi negatif:")
    print(negative_correlations)

#4
data = dataWalmart[['Fuel_Price', 'Weekly_Sales']]

X = data[['Fuel_Price']]
y = data['Weekly_Sales']

model = LinearRegression()
model.fit(X, y)

a = model.intercept_
b = model.coef_[0]

print("Model regresi: y = {} + {}x".format(a, b))
data = dataWalmart[['Fuel_Price', 'Weekly_Sales']]

X = data[['Fuel_Price']]
y = data['Weekly_Sales']

model = LinearRegression()

model.fit(X, y)
y_pred = model.predict(X)
plt.scatter(X, y, color='blue', label='Data')
plt.plot(X, y_pred, color='red', linewidth=2, label='Regression Line')

plt.xlabel('Fuel_Price')
plt.ylabel('Weekly_Sales')
plt.title('Linear Regression')

plt.legend()

plt.show()




















