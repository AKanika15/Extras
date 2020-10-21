# Regular libraries for analysis of data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Models for sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Models for evaluation
from sklearn.model_selection import cross_val_score, train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import confusion_matrix,classification_report,f1_score,recall_score,precision_score,roc_curve,plot_roc_curve
data=pd.read_csv("churnPrediction.csv")
data.head()
data.shape
df=pd.DataFrame(data)
df.head()
df["occupation"].count()
df["churn"].value_counts()
df["churn"].value_counts().plot(kind="bar",color=["r","b"]);
df.info()
df.isna().sum()
df.describe()
# Split into X and y
X=df.drop("churn",axis=1)
y=df["churn"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

# Fill the categories using scikit
cat_imputer=SimpleImputer(strategy="constant",fill_value="missing")
num_imputer=SimpleImputer(strategy="mean")

#Defining columns
gender_occupation=["gender","occupation","city"]
num_features=["dependents","days_since_last_transaction"]

#Create an imputer(something that fills the data)
imputer=ColumnTransformer([
    ("cat_imputer",cat_imputer,gender_occupation),
    ("num_imputer",num_imputer,num_features)
        ])

# Transforming the data
filled_X=imputer.fit_transform(X)
filled_X
churn_filled=pd.DataFrame(filled_X,columns=["gender","occupation","city","dependents","days_since_last_transaction"])
churn_filled.head(10)
churn_filled.isna().sum()
df["gender"]=churn_filled["gender"]
df["gender"].isna().sum()
df["occupation"]=churn_filled["occupation"].astype(str)
df["city"]=churn_filled["city"].astype(str)
df["dependents"]=churn_filled["dependents"].astype(str)
df["days_since_last_transaction"]=churn_filled["days_since_last_transaction"].astype(str)
df.isna().sum()
# Split into X and y
X=df.drop("churn",axis=1)
y=df["churn"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# Let's convert the data to numbers

one_hot=OneHotEncoder()
categories=['gender', 'dependents', 'occupation',
       'city','days_since_last_transaction']
transformer=ColumnTransformer([("one_hot",
                                  one_hot,
                                   categories)],
                             remainder="passthrough")

transformed_X=transformer.fit_transform(X)
categorial_features=['gender']
one_hot=OneHotEncoder
column_transformer= ColumnTransformer([("one_hot",one_hot,categorial_features)],remainder="passthrough")

transformed_X=transformer.fit_transform(df)
transformed_X
pd.DataFrame(transformed_X)
np.random.seed(11)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(transformed_X,y,test_size=0.3)

# Out the models in one place e.g. Dictionary
models={"LR":LogisticRegression(),
        "RFC":RandomForestClassifier(n_estimators=100),
        "Knn":KNeighborsClassifier() }

# Create a function to fit and score models

def fit_score(model,X_train,y_train,X_test,y_test):
    m_score={} #to keep the model score as dictionary
    
    #Loop for models
    for name,m in model.items():
        m.fit(X_train,y_train)
        m_score[name]=m.score(X_test,y_test)
    return m_score

np.random.seed(42)
model_score=fit_score(model=models,X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test);
print(model_score)
model_compare=pd.DataFrame(model_score,index=["models_name"])
model_compare.plot.bar();