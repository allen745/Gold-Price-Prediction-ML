# data collection --> data pre processing--> data analysis --> train test data --> Random forest regression --> evaluation

# importing dependencies

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# data collection and processing
# loading csv data to pandas dataFrame

gold_data = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\gld_price_data.csv')
print(gold_data.head()) # first 5 rows
print(gold_data.tail()) # last 5 rows
print(gold_data.shape) # shape of dataset
print(gold_data.describe()) # give static info
print(gold_data.info()) # give information about dataset

# checking for nullvalues
print(gold_data.isnull().sum())

# correlation
# positive correlation
# negative correlation
correlation = gold_data.corr(numeric_only=True)
print(correlation)

# construct a heat map to understand the correlation

plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, annot=True, square=True, fmt='.1f',annot_kws={'size':8}, cmap='Blues')
plt.show()

print(correlation['GLD'])

# checking the distribution of GLD price
sns.displot(gold_data['GLD'], color='blue')
plt.show()

# splitting the data in to future and target

x = gold_data.drop(['Date','GLD'] ,axis=1)
y = gold_data['GLD']
print(x)
print(y)

# splitting into test data and train data

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)


# modal evalution
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x_train,y_train)

# prediction on test data
test_data_prediction = regressor.predict(x_test)
# print(test_data_prediction)

# R square error

error_score = metrics.r2_score(y_test, test_data_prediction)
print("R Squared Error: ",error_score)

# compare actual values and predicted values
y_test = list(y_test)

plt.plot(y_test,color='red', label='Actual Values')
plt.plot(test_data_prediction, color='green', label='Predicted Values')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.show()

spx = float(input("Enter SPX Value: "))
uso = float(input("Enter USO Value: "))
slv = float(input("Enter SLV Value: "))
eur_usd = float(input("Enter EUR/USD Value: "))

input_data = np.array([[spx, uso, slv, eur_usd]])

prediction = regressor.predict(input_data)

print("\nPredicted Gold Price (GLD): $", round(prediction[0], 2))

print("\nPredicted Gold Price (GLD): ${:.2f}".format(prediction[0]))

# Confidence message
if error_score > 0.95:
    print("Model Status: Excellent Accuracy")
elif error_score > 0.85:
    print("Model Status: Good Accuracy")
else:
    print("Model Status: Needs Improvement")