# car data--> data pre processing --> Train test split --> Linear & lasso Regression modal
# importing dependencies

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

# data collection and processing
car_dataset = pd.read_csv(r'C:\Users\allen\OneDrive\Desktop\understanding\data\car data\car data.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(car_dataset.head())
print(car_dataset.shape)

# getting some information about dataset
print(car_dataset.info())

# static data
print(car_dataset.describe())

# checking null dta
print(car_dataset.isnull().sum())

# checking for  distribution of categorical data
print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())

# Encoding the Categorical data
# encode "Fuel_Type" Column .
car_dataset.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}}, inplace=True)


# encode "Seller_Type" Column .
car_dataset.replace({'Seller_Type':{'Dealer':0,'Individual':1}}, inplace=True)


# encode "Transmission" Column .
car_dataset.replace({'Transmission':{'Manual':0,'Automatic':1}}, inplace=True)

print(car_dataset.head())

x = car_dataset.drop(['Car_Name','Selling_Price'],axis=1)
y = car_dataset['Selling_Price']
print(x)

# SPLITTING DATA IN TO TRAIN AND TEST DATA
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)

# modal training

linreg = LinearRegression()
linreg.fit(x_train,y_train)

# modal evaluation
training_data_prediction = linreg.predict(x_train)

# R square Error

error_score = metrics.r2_score(y_train,training_data_prediction)
print("R square Error: ",error_score)

# visulaise the actual price and predicted price
plt.scatter(y_train,training_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

test_data_prediction = linreg.predict(x_test)
error_score = metrics.r2_score(y_test,test_data_prediction)
print("R square Error: ",error_score)

plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Price")

plt.scatter(y_test,test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


# lasso regression
lasso = Lasso()
lasso.fit(x_train,y_train)

training_data_prediction2 = lasso.predict(x_train)
error_score = metrics.r2_score(y_train,training_data_prediction2)
print("R square Error: ",error_score)

plt.scatter(y_train,training_data_prediction2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

test_data_prediction2 = lasso.predict(x_test)
error_score = metrics.r2_score(y_test,test_data_prediction2)
print("R square Error: ",error_score)
plt.scatter(y_test,test_data_prediction2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()





