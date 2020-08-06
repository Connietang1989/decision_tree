# decision_tree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from IPython.display import Image  
from sklearn import metrics #Import scikit-learn metrics module for accuracy
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.externals.six import StringIO 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from subprocess import call

######################  import CSV file ##########################################################

adult = pd.read_csv(r"/Users/connietang/Desktop/adult.csv")
adult.head()
df = pd.read_csv(r"/Users/connietang/Desktop/adult.csv")
test = pd.read_csv(r"/Users/connietang/Desktop/test_adult.csv")
test.head()
df_test = pd.read_csv(r"/Users/connietang/Desktop/test_adult.csv")

###################### Generate data overview for further analysis ###############################

#  Step 2 Generate data overview for further analysis
print ("Rows     : " ,adult.shape[0])
print ("Columns  : " ,adult.shape[1])
print ("\nFeatures : \n" ,adult.columns.tolist())
print ("\nMissing values :  ", adult.isnull().sum().values.sum())
print ("\nUnique values :  \n",adult.nunique())
print(adult.head(20))
pd.set_option("display.float_format", "{:.2f}".format)
print(adult.describe())
print(adult.info())
sns.pairplot(df)
plt.figure(figsize=(40,40))
plt.figure(figsize=(30,30))
sns.heatmap(df.corr(), annot = True,fmt='.1g',vmin=-1, vmax=1, center= 0,square=True)


##############    Data Preprocessing   #########################################################

## Step 3 Handling missing values
Id_col = ['id']
target_col = ["income"]
#replace values
adult= adult.replace('[?]', np.nan, regex=True)
test= test.replace('[?]', np.nan, regex=True)
adult.fillna("NaN",inplace=True)
test.fillna("NaN",inplace=True)
adult.select_dtypes(include='object').fillna(\
adult.select_dtypes(include='object').mode().iloc[0])
test.select_dtypes(include='object').fillna(\
test.select_dtypes(include='object').mode().iloc[0])
total = adult.isnull().sum().sort_values(ascending=False)
percent = (adult.isnull().sum()/adult.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
f, ax = plt.subplots(figsize=(15, 6))
plt.xticks(rotation='90')
sns.barplot(x=missing_data.index, y=missing_data['Percent'])
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
missing_data.head()


## Step 4 Label encoding cat. features
obj_df_adult = adult.select_dtypes(include=['object']).copy()
obj_df_adult.head()
le=LabelEncoder()
for i in obj_df_adult:
    adult[i]=le.fit_transform(adult[i])

obj_df_test = test.select_dtypes(include=['object']).copy()
obj_df_test.head()
le=LabelEncoder()
for i in obj_df_test:
    test[i]=le.fit_transform(test[i])
  
## mean removal    
data_standardized = preprocessing.scale(adult)
data_standardized = preprocessing.scale(test)

## Step 5 Scaling Numerical columns
num_cols = [x for x in adult.columns if x not in target_col + Id_col]
std = StandardScaler()
scaled = std.fit_transform(adult[num_cols])
scaled = pd.DataFrame(scaled, columns=num_cols)

# dropping original values merging scaled values for numerical columns
df_adult_og = adult.copy()
adult = adult.drop(columns=num_cols, axis=1)
adult = adult.merge(scaled, left_index=True, right_index=True, how="left")
summary = (df_adult_og[[i for i in df_adult_og.columns if i not in Id_col]].
           describe().transpose().reset_index())
summary = summary.rename(columns={"index": "feature"})
summary = np.around(summary, 3)
val_lst = [summary['feature'], summary['count'],
           summary['mean'], summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]

num_test_cols = [x for x in test.columns if x not in target_col + Id_col]
std = StandardScaler()
scaled = std.fit_transform(test[num_test_cols])
scaled = pd.DataFrame(scaled, columns=num_test_cols)
print (summary)

# dropping original values merging scaled values for numerical columns
df_test_og = test.copy()
test = test.drop(columns=num_test_cols, axis=1)
test= test.merge(scaled, left_index=True, right_index=True, how="left")
summary = (df_test_og[[i for i in df_test_og.columns if i not in Id_col]].
           describe().transpose().reset_index())
summary = summary.rename(columns={"index": "feature"})
summary = np.around(summary, 3)
val_lst = [summary['feature'], summary['count'],
           summary['mean'], summary['std'],
           summary['min'], summary['25%'],
           summary['50%'], summary['75%'], summary['max']]
print (summary)



#####################  splitting train and test data ############################################## 
## Step 6 Define dependent and independent variables
cols    = [i for i in adult.columns if i not in Id_col + target_col]
cols_test    = [i for i in test.columns if i not in Id_col + target_col]
train_X = adult[cols]
train_Y = adult[target_col]
test_X  = test[cols_test]
test_Y  = test[target_col]

################### Random Forest classification,features for tree based models ####################################

# Building random forest classification model
clf = RandomForestClassifier(n_estimators=200, random_state=0,max_depth=4)
clf.fit(train_X, train_Y)
y_pred = clf.predict(test_X)

# Model Accuracy, how often is the classifier correct?
print(confusion_matrix(test_Y,y_pred))
print(classification_report(test_Y,y_pred))
print(accuracy_score(test_Y, y_pred))
print("Accuracy:",metrics.accuracy_score(test_Y, y_pred))

# Visualize the most importance features
feature_imp = pd.Series(clf.feature_importances_,index=cols).sort_values(ascending=False)
feature_imp
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()

# Extract single tree
estimator = clf.estimators_[5]

#  Plotting decision tree from random forest
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = cols,
                class_names = [' <=50K','>50K'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)
       
# Convert to png using system command (requires Graphviz)
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
Image(filename = 'tree.png')
