
import pandas as pd

cement=pd.read_csv('https://github.com/ybifoundation/Dataset/raw/main/Concrete%20Compressive%20Strength.csv')

cement.head()

cement.describe()

cement.isnull().sum()

cement.nunique()

import seaborn
seaborn.pairplot(cement)

cement.columns

y=cement['Concrete Compressive Strength(MPa, megapascals) ']

x=cement[['Cement (kg in a m^3 mixture)',
       'Blast Furnace Slag (kg in a m^3 mixture)',
       'Fly Ash (kg in a m^3 mixture)', 'Water (kg in a m^3 mixture)',
       'Superplasticizer (kg in a m^3 mixture)',
       'Coarse Aggregate (kg in a m^3 mixture)',
       'Fine Aggregate (kg in a m^3 mixture)', 'Age (day)',]]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=2529)

x_train.shape,x_test.shape,y_train.shape,y_test.shape

from sklearn.linear_model import LinearRegression
model=LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error,mean_squared_error

mean_absolute_error(y_test,y_pred)

mean_absolute_percentage_error(y_test,y_pred)

mean_squared_error(y_test,y_pred)

cement.sample()

x_new=x.sample()
x_new

model.predict(x_new)

