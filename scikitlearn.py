import pandas as pd
import numpy as np
heart_disease=pd.read_csv("heart-disease.csv.csv")
heart_disease
from sklearn.svm import LinearSVC
np.random.seed(42)
x=heart_disease.drop("target",axis=1)
y=heart_disease["target"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=LinearSVC(max_iter=1000)
clf.fit(x_train,y_train)

LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
          intercept_scaling=1, loss='squared_hinge', max_iter=1000,
          multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
          verbose=0)
clf.score(x_test,y_test)

from sklearn.ensemble import RandomForestClassifier
np.random.seed(42)
x=heart_disease.drop("target",axis=1)
y=heart_disease["target"]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
clf=RandomForestClassifier()
clf.fit(x_train,y_train)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
clf.score(x_test,y_test)
clf.predict(x_test)

np.array([y_test])

y_preds=clf.predict(x_test)
np.mean(y_preds==y_test)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_preds)

clf.predict_proba(x_test[:5])

heart_disease.dtypes

x=heart_disease.drop("target",axis=1)
y=heart_disease["target"]
x


y

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()
clf.get_params
​
<bound method BaseEstimator.get_params of RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)>
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape , y_train.shape
len(heart_disease)
clf.fit(x_train,y_train)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=100,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
x
from sklearn.metrics import roc_curve
​
# Fit the classifier
clf.fit(x_train, y_train)
​
# Make predictions with probabilities
y_probs = clf.predict_proba(x_test)
​
y_probs[:10], len(y_probs)

y_probs_positive = y_probs[:, 1]
y_probs_positive[:10]

​
# Caculate fpr, tpr and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_probs_positive)
​
# Check the false positive rates
thresholds

import matplotlib.pyplot as plt
def plot_roc_curve(fpr,tpr):
    """
    Plots a ROC curve
    """
    plt.plot(fpr,tpr)
    plt.show()
plot_roc_curve(fpr,tpr)    

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_probs_positive)

y_label=clf.predict(x_test)
y_label

clf.score(x_train,y_train)

clf.score(x_test,y_test)

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
print(classification_report(y_test,y_label))
            
confusion_matrix(y_test,y_label)

pd.crosstab(y_test,y_label,rownames=["Actual label"],colnames=["predicted label"])

import seaborn as sns
sns.set(font_scale=1.5)
conf_mat=confusion_matrix(y_test,y_label)
sns.heatmap(conf_mat)


def plot_conf_mat(conf_mat):
    """
     Plots a confusion matrix using Seaborn's heatmap().
    """
    fig,ax=plt.subplots(figsize=(3,3))
    ax=sns.heatmap(conf_mat,annot=True,cbar=False)
plot_conf_mat(conf_mat)    

x=heart_disease.drop("target",axis=1)
y=heart_disease["target"]
x.head()

from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(clf,x,y)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_label))
         

def plot_conf_mat(conf_mat):
    """
    Plots a confusion matrix using Seaborn's heatmap().
    """
    fig, ax = plt.subplots(figsize=(3,3))
    ax = sns.heatmap(conf_mat,
                     annot=True, # Annotate the boxes with conf_mat info
                     cbar=False)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    
    # Fix the broken annotations (this happened in Matplotlib 3.1.1)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top-0.5);
    
plot_conf_mat(conf_mat)

accuracy_score(y_test,y_label)

np.random.seed(42)
for i in range(10, 100, 10):
    print(f"Trying model with {i} estimators...")
    clf = RandomForestClassifier(n_estimators=i).fit(x_train, y_train)
    print(f"Model accuracy on test set: {clf.score(x_test, y_test) * 100:.2f}%")
    print("")
import pickle
pickle.dump(clf,open("random_model.pk1","wb"))
loaded_model=pickle.load(open("random_model.pk1","rb"))
loaded_model.score(x_test,y_test)