import pandas as pd
import numpy as np
import category_encoders as ce
import sys, math, pickle
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer

data = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")

le = LabelEncoder()
ohc = OneHotEncoder()

data = data[["Year of Record","Gender","Age","Country","Size of City","Profession","University Degree","Body Height [cm]", "Income in EUR"]]

data["Age"].fillna(method="ffill", inplace= True)
data["Profession"].fillna(method="ffill", inplace= True)
data["Year of Record"].fillna(method="ffill", inplace= True)
data["Gender"].fillna(method="ffill", inplace= True)

data = data.replace('unknown', np.nan)
data = data.dropna(how = 'any')

y = data['Income in EUR']

data = pd.get_dummies(data, prefix_sep='_', drop_first=True)

data = data.apply(le.fit_transform)

transformer = QuantileTransformer(output_distribution='normal')

X = (data.drop(["Income in EUR"],1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LinearRegression()

ttr = TransformedTargetRegressor(regressor=clf,transformer=transformer)
ttr.fit(X_train, y_train)

y_pred = ttr.predict(X_test)

print("\nRoot Mean Squared Error: ", math.sqrt(mean_squared_error(y_test, y_pred)))



filename = "clf.sav"
pickle.dump(ttr, open(filename,"wb"))

data2 = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")

data2 = data2[["Year of Record","Gender","Age","Country","Size of City","Profession","University Degree","Body Height [cm]"]]

data2["Age"].fillna(method="ffill", inplace= True)
data2["Profession"].fillna(method="ffill", inplace= True)
data2["Year of Record"].fillna(method="ffill", inplace= True)
data2["Gender"].fillna(method="ffill", inplace= True)

data2 = data2.replace('unknown', np.nan)

data2 = pd.get_dummies(data2, prefix_sep='_', drop_first=True)
data2 = data2.apply(le.fit_transform)

X, data2 = X.align(data2, join='left', axis=1)

data2.fillna(0,inplace= True)

loaded_model = pickle.load(open('clf.sav', 'rb'))
result = loaded_model.predict(data2)
np.savetxt('test.csv', result, delimiter=',')
