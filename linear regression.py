import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse
experince=[1,1.5,2,3.4,4.6,5,5.5,6.2,7,8.9]
salary=[2,2.3,3.6,4.9,5.6,6,7.2,8,9.7,10]
data=pd.DataFrame({
    "experince":experince,
    "salary":salary
})
data
plt.scatter(data.experince,data.salary)
beta=1.5
b=1.1
line1=[]
for i in range(len(data)):
    line1.append(data.experince [i] * beta+b)
plt.scatter(data.experince,data.salary,color="red")
plt.plot(data.experince,line1,color="black",label="line")
MSE=mse(data.experince,line1)
MSE
def Error(beta,data):
    b=1.1
    salary=[]
    experince=data.experince
    for i in range(len(data.experince)):
        tmp=data.experince*beta+b
        salary.append(tmp)
    MSE=mse(experince,salary)
    return MSE
train=pd.read_csv("train_cleaned.csv")
train.head()
x=train.drop(["Item_Outlet_Sales"],axis=1)
y=train["Item_Outlet_Sales"]
x=train.drop(["Item_Outlet_Sales"],axis=1)
y=train["Item_Outlet_Sales"]
rom sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import mean_absolute_error as mae
Lr=lr(normalize=True)
Lr.fit(x_train,y_train)
train_predict=Lr.predict(x_train)
k=mae(train_predict,y_train)
k
test_predict=Lr.predict(x_test)
k=mae(test_predict,y_test)
k
Lr.coef_
plt.figure()
x=range(len(x_train.columns))
y=Lr.coef_
plt.bar(x,y)