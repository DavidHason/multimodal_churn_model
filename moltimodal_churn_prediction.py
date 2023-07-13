#!/usr/bin/env python
# coding: utf-8

# # Multi-Model Learning

# #### Author - David

# ---

# In[1]:


import os
import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)

import warnings
warnings.filterwarnings('ignore')

# Matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
from matplotlib.colorbar import ColorbarBase

# Scipy
from scipy import interpolate
from scipy import spatial
from scipy import stats
from scipy.cluster import hierarchy
# Others
import numpy as np
import pandas as pd
import datetime as dt
import seaborn as sns
import pickle
from math import modf

from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix #for confusion matrix
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from sklearn import metrics
import xgboost as xgb
from sklearn.linear_model import SGDClassifier


sns.set(style='white', context='notebook', palette='deep')
from sklearn.model_selection import train_test_split, KFold, cross_validate

#Models
import warnings
warnings.filterwarnings("ignore")


from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import StratifiedShuffleSplit, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, precision_recall_fscore_support, roc_auc_score
#from sklearn.inspection import plot_partial_dependence


import glob
import os
import librosa
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import soundfile as sf
#from sklearn.externals import joblib
from tensorflow import keras 
from tensorflow.keras.preprocessing import image
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image


#from sklearn.externals import joblib
import sklearn
#from sklearn.externals import joblib
import joblib
import tensorflow.keras as tf
from tensorflow.keras import models
from tensorflow.keras import layers
#from python_utils import model_evaluation_utils as meu
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import h5py
from tensorflow.keras.models import load_model
import h5py

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

from platform import python_version

print(python_version())

import tensorflow

from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
import numpy as np
import sys
from tensorflow.keras.models import load_model
#import model_evaluation_utils
#import model_evaluation_utils as meu

import sklearn
#from sklearn.externals import joblib
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras
import tensorflow.keras as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # I believe this is better optimizer for our case
from tensorflow.keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
from tensorflow.keras.models import load_model
#import model_evaluation_utils
#import model_evaluation_utils as meu

import json
import warnings

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.util.tf_export import keras_export

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils


from tensorflow import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image

def process_sound_data(data):
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    return data

#from tensorflow.keras_preprocessing.image import ImageDataGenerator#from tensorflow.keras.preprocessing import image_dataset_from_directory
#from tensorflow.keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier  
from sklearn.exceptions import ConvergenceWarning

pd.options.display.float_format = '{:.5f}'.format
import numpy as np
import matplotlib.pyplot as plt
plt.style.context('seaborn-talk')
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': '16',
          'figure.figsize': (15, 5),
         'axes.labelsize': '20',
         'axes.titlesize':'30',
         'xtick.labelsize':'18',
         'ytick.labelsize':'18'}
plt.rcParams.update(params)

plt.rcParams['text.color'] = '#A04000'
plt.rcParams['xtick.color'] = '#283747'
plt.rcParams['ytick.color'] = '#808000'
plt.rcParams['axes.labelcolor'] = '#283747'

# Seaborn style (figures)
sns.set(context='notebook', style='whitegrid')
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from sklearn import metrics

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score


# ### Define Helper Function

# In[2]:


def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])
]


def remove_collinear_features(x, threshold = 0.99):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return x

# One-hot encoding
def one_hot_encoding(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(each, 1)
    return df

# Function to min-max normalize
def normalize(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode
    @return a DataFrame with normalized specified features
    """
    result = df.copy() # do not touch the original df
    for feature_name in cols:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        if max_value > min_value:
            result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def summarize_categoricals(df, show_levels=False):
    """
        Display uniqueness in each column
    """
    data = [[df[c].unique(), len(df[c].unique()), df[c].isnull().sum()] for c in df.columns]
    df_temp = pd.DataFrame(data, index=df.columns,
                           columns=['Levels', 'No. of Levels', 'No. of Missing Values'])
    return df_temp.iloc[:, 0 if show_levels else 1:]


def find_categorical(df, cutoff=10):
    """
        Function to find categorical columns in the dataframe.
    """
    cat_cols = []
    for col in df.columns:
        if len(df[col].unique()) <= cutoff:
            cat_cols.append(col)
    return cat_cols


def to_categorical(columns, df):
    """
        Converts the columns passed in `columns` to categorical datatype
    """
    for col in columns:
        df[col] = df[col].astype('category')
    return df


# # Performance Assessment

# # Step #1 - Prediction of Financial Literacy and 1st Fusion Variable

# In[3]:


# Load data
final_df_fl = pd.read_csv('Multimodal_Databases/Financial Literacy Dataset/FL_dataset.csv')
# Drop the unnecessary features
final_df_fl = final_df_fl.drop(['EMAIL_CHANGE','TITLE_CHANGE','HOME_TEL_CHANGE','MOBILE_CHANGE','WORK_TEL_CHANGE','spouse_amt','Direct to Direct','Investor_Investigation','Investor_Face to face','NewAdviser','NewDealer','Non-member_Call Transfer', 'Investor_Distribution', 'Investor_Internal enquiry','Investor_Platinum Adviser','Non-member_Face to face','Non-member_Internal enquiry','Non-member_Platinum Adviser','ONLINE-Change Address','ONLINE-New Funded Applications - OAF','Personal_inj','Small_Bus_15yr_Ex','Small_Bus_Ret_Ex','personal_inj_amt','Small_Bus_Ret_Ex_amt','Small_Bus_15yr_Ex_amt','Non-member_Email','ONLINE-Additional Applications', 'ONLINE-RIP Maintenance','ONLINE-Unfunded Applications - OAF','Spouse_non_conc', 'CUST_GIVEN_CHANGE','CUST_NAME_CHANGE','Direct_to_Advised','Non-member_Investigation','Non-member_Inbound'], axis=1)
final_df_fl.head()


# In[4]:


categoricals = final_df_fl.select_dtypes(include='int64').columns.to_list()
summarize_categoricals(final_df_fl[categoricals], show_levels=True)


# In[5]:


categoricals = summarize_categoricals(final_df_fl[categoricals], show_levels=True)

categoricals_fl = categoricals.index.tolist()


# In[6]:


# Extract categorical variables
categoricals_fl.remove('new_id')
categoricals_fl.remove('FL')

remove_cols = ['FNILogin', 'Investor_Inbound', 'ONLINE-Switches', 'Personal_Non_conc', 'salary_Sacr', 'super_guarantee', 
               'Switch', 'Withdrawal']
categoricals_fl = list(set(categoricals_fl) - set(remove_cols))


# In[7]:


# get the intial set of encoded features and encode them
final_df_fl = one_hot_encoding(final_df_fl, categoricals_fl)

print('Dimensions of the Training set:',final_df_fl.shape)
numeric_cols_fl = ['Financial_Literacy', 'Change_12m_Cash', 'Change_12m_Fixed_interest', 'Change_12m_Global_fixed_interest',
'Change_12m_Australian_share', 'Change_12m_Global_share', 'Change_12m_Property', 'ADDRESS_CHANGE', 'FNILogin', 
'Investor_Call Transfer', 'Investor_Email', 'Investor_Inbound', 'Investor_Outbound', 'Non-member_Outbound',
'ONLINE-Switches', 'Personal_conc', 'Personal_Non_conc', 'Rollover', 'Rollover_wdl', 'salary_Sacr', 'super_guarantee', 
'Switch', 'Withdrawal', 'personal_non_conc_amt', 'personal_conc_amt', 'salary_sacr_amt', 'Gov_co_cont_amt', 'SG_amt', 
'Total_concessional', 'Total_non_concessional', 'concessional_cap']

numeric_cols_fl = list(set(numeric_cols_fl) - set(categoricals_fl))

### Splitting into features and labels dataframe

final_df_fl = final_df_fl.dropna()
final_df_fl.reset_index(inplace = True, drop = True)

from sklearn.preprocessing import StandardScaler
scaler_fl = StandardScaler()
final_df_fl.loc[:,numeric_cols_fl] = scaler_fl.fit_transform(final_df_fl.loc[:,numeric_cols_fl])

final_df_fl['FL'] = final_df_fl['FL'].astype('object')
final_df_fl.head()


# In[8]:


fl_ids = final_df_fl['new_id']


# In[9]:


finalDF = final_df_fl.copy()
ids = finalDF['new_id']
finalDF['FL'] = finalDF['FL'].astype('object')
x = finalDF.drop(['FL', 'new_id', 'Financial_Literacy'], axis = 1)
y = finalDF['FL']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)


# # Comparison Table of ML based Models - Train Test Split

# In[10]:


from sklearn.metrics import cohen_kappa_score
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, shuffle=True)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train = X_train.fillna(method='ffill')
X_test = X_test.fillna(method='ffill')


# In[11]:


print('Shape of Training Features : ', X_train.shape)
print('Shape of Testing Features : ', X_test.shape)

print('Shape of Training Labels : ', y_train.shape)
print('Shape of Testing Labels : ', y_test.shape)


# In[12]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
Y_train = y_train.copy()
print('Train data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))


# In[13]:


X_train_d5 = X_train.copy()
X_test_d5 = X_test.copy()
y_train_d5 = y_train.copy() 
y_test_d5 =  y_test.copy()


# In[14]:


get_ipython().run_line_magic('matplotlib', 'inline')
train_accuracy = []
test_accuracy = []
precision = []
recall = []
f1 = []
cohen_kappa = []
models = ["Naive Bayes","Logistic Regression","Decision Tree","RandomForest", "AdaBoost", "ExtraTrees","GradientBoosting","XGboost"]
roc = []
mathew = []
random_state = 2
classifiers = []
classifiers.append(BernoulliNB())
classifiers.append(LogisticRegression())
classifiers.append(DecisionTreeClassifier())
classifiers.append(RandomForestClassifier(random_state=random_state, max_depth = 10, max_features = 'sqrt', n_estimators=  300))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.5))
classifiers.append(ExtraTreesClassifier(random_state=random_state, criterion ='entropy', max_features = 'sqrt', min_samples_leaf = 20, min_samples_split = 15))
classifiers.append(GradientBoostingClassifier(random_state=random_state, learning_rate = 0.2, max_depth = 10, n_estimators = 200))
classifiers.append(XGBClassifier(random_state = random_state))



for classifier,model in zip(classifiers, models):
    print('='*len(model))
    print(model)
    print('='*len(model))
    classifier.fit(X_train, y_train)
    trainprediction = classifier.predict(X_train)
    prediction = classifier.predict(X_test)
    trainaccuracy = accuracy_score(y_train, trainprediction)
    testaccuracy = accuracy_score(y_test, prediction)
    train_accuracy.append(trainaccuracy)
    test_accuracy.append(testaccuracy)
    precision.append(precision_score(y_test, prediction, average='macro'))
    recall.append(recall_score(y_test, prediction, average='macro'))
    cohen_kappa.append(cohen_kappa_score(y_test, prediction))
    f1.append(f1_score(y_test, prediction, average='macro'))
    roc.append(metrics.roc_auc_score(y_test, prediction))

    mathew.append(metrics.matthews_corrcoef(y_test, prediction))

    print('\n clasification report:\n', classification_report(y_test,prediction))
    print('\n confussion matrix:\n',confusion_matrix(y_test, prediction))
    print('\n')
    
scoreDF = pd.DataFrame({'Model' : models})
scoreDF['Train Accuracy'] = train_accuracy
scoreDF['Test Accuracy'] = test_accuracy
scoreDF['Precision'] =  precision
scoreDF['Recall'] =  recall
scoreDF['F1 Score'] = f1 
scoreDF['AUC Score'] = roc 
scoreDF['Matthew Correlation Coefficient'] = mathew
scoreDF['Cohen Kappa Score'] = cohen_kappa


scoreDF.set_index("Model")


# In[15]:


plt.style.use('fivethirtyeight')
params = {'legend.fontsize': '16',
          'figure.figsize': (15, 5),
         'axes.labelsize': '20',
         'axes.titlesize':'30',
         'xtick.labelsize':'18',
         'ytick.labelsize':'18'}
plt.rcParams.update(params)

def subcategorybar(X, vals, width=0.8):
    cols = ['Train Accuracy', 'Test Accuracy', 'Precision', 'AUC Score','Recall','F1 Score', 'Cohen Kappa Score']
    n = len(vals)
    _X = np.arange(len(X))
    for i in range(n):
        plt.bar(_X - width/2. + i/float(n)*width, vals[i], width=width/float(n), align="edge")   
        
        
    plt.xticks(_X, X)
    
plt.figure(figsize = (17,6))
subcategorybar(models, [scoreDF['Train Accuracy'], scoreDF['Test Accuracy'], scoreDF['Precision'], scoreDF['AUC Score'], scoreDF['Recall'], scoreDF['F1 Score'], scoreDF['Cohen Kappa Score']])
plt.ylim(0, 1.0)
cols = ['Train Accuracy', 'Test Accuracy', 'Precision', 'AUC','Recall','F1 Score', 'Cohen Kappa Score']
#plt.legend(cols)

plt.legend(cols, loc='upper center', bbox_to_anchor=(0.5, -0.35),
          fancybox=True, shadow=True, ncol=7)


plt.xlabel('Model', fontsize = 20)
plt.xticks(rotation = 45)

plt.title("Comparison of Models", fontsize = 24)
plt.show();


# In[16]:


scoreDF = scoreDF.set_index("Model")
scoreDF


# In[17]:


ax = scoreDF['Test Accuracy'].round(6).plot(kind = 'bar', color = 'orange');

for p in ax.patches:
    ax.annotate(str(p.get_height()), (p.get_x() * 1., p.get_height() * 0.95))
plt.title('Test Accuracy', fontsize = 18)
plt.xticks(rotation = 60)
plt.show();


# In[18]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import learning_curve

# Define the XGBoost model
model = xgb.XGBClassifier()

# Define the hyperparameter grid
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'min_child_weight': [1, 3, 5]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Make predictions on the test data using the best model
# Make predictions on the test data using the best model
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]  # Use the predicted probabilities for the positive class
cm = confusion_matrix(y_test, predictions)
# Evaluate the best model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

# Print the evaluation metrics
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Confusion matrix subplot
axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axs[0].set_title("Confusion Matrix", fontsize = 24)
tick_marks = np.arange(len(np.unique(y_test)))
axs[0].set_xticks(tick_marks)
axs[0].set_yticks(tick_marks)
axs[0].set_xticklabels(np.unique(y_test))
axs[0].set_yticklabels(np.unique(y_test))
axs[0].set_xlabel('Predicted Label')
axs[0].set_ylabel('True Label')
for i in range(len(np.unique(y_test))):
    for j in range(len(np.unique(y_test))):
        axs[0].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

        
# Learning curve subplot
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

axs[1].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
axs[1].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')
axs[1].set_xlabel('Training Set Size')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Learning Curve', fontsize=24)
axs[1].legend(loc='lower right', facecolor='white')


# ROC curve subplot
axs[2].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
axs[2].plot([0, 1], [0, 1], color='red', linestyle='--')
axs[2].set_xlim([0.0, 1.0])
axs[2].set_ylim([0.0, 1.05])
axs[2].set_xlabel('False Positive Rate')
axs[2].set_ylabel('True Positive Rate')
axs[2].set_title('Receiver Operating Characteristic', fontsize = 24)
axs[2].legend(loc='lower right')

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('model_evaluation_FL_Model.png')
plt.suptitle('FL Model Performance', fontsize = 28, y = 1.1, color = 'red')
# Show the figure
plt.show();


# In[19]:


fl_model = best_model
fl_predictions = best_model.predict(x)
fl_modelled_data = pd.DataFrame({'new_id': fl_ids, 'fl_predictions':fl_predictions, 'ground_truths': y})
fl_modelled_data.head()


# In[20]:


pd.crosstab(fl_modelled_data['fl_predictions'], fl_modelled_data['ground_truths'])


# # Step #2 Emotion Prediction and Fusion Feature
# 
# ##### For emotion recognition model do:
# 
# Assign number 1 for all emotions predicted as Anger and Sadness and also assign number 0 for emotions predicted as Happiness and boredom. And then add this binary output to the Churn Dataset.

# In[21]:


import glob
import os
import librosa
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import soundfile as sf
#from sklearn.externals import joblib
from tensorflow import keras 
from tensorflow.keras.preprocessing import image
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf
#from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing import image

#from sklearn.externals import joblib
import sklearn
#from sklearn.externals import joblib
import joblib
import tensorflow.keras as tf
from tensorflow.keras import models
from tensorflow.keras import layers
#from python_utils import model_evaluation_utils as meu
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

import h5py
from tensorflow.keras.models import load_model
import h5py

import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
print(tf.test.is_built_with_cuda())

from platform import python_version

print(python_version())

import tensorflow

from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.preprocessing import label_binarize
from scipy import interp
from sklearn.metrics import roc_curve, auc 
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Input, Flatten, Dense, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras import regularizers
import numpy as np
import sys
from tensorflow.keras.models import load_model
#import model_evaluation_utils
#import model_evaluation_utils as meu

import sklearn
#from sklearn.externals import joblib
import joblib

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow.keras
import tensorflow.keras as tf
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scipy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D,AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import LearningRateScheduler,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam # I believe this is better optimizer for our case
from tensorflow.keras.preprocessing.image import ImageDataGenerator # to augmenting our images for increasing accuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
get_ipython().run_line_magic('matplotlib', 'inline')
import sys
from tensorflow.keras.models import load_model
#import model_evaluation_utils
#import model_evaluation_utils as meu

import json
import warnings

import numpy as np

from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow import keras 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from PIL import Image

def process_sound_data(data):
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    return data


# In[22]:


import glob

ROOT_DIR = 'Multimodal_Databases/Customer_sentiment_Dataset/voice_files/'
files = glob.glob(ROOT_DIR + '*')
len(files)


# In[23]:


def get_sound_data(path, sr=44100):
    data, fsr = sf.read(path) #read sound file
    data_resample = librosa.resample(data.T, fsr, sr)
    if len(data_resample.shape) > 1:
        data_resample = np.average(data_resample, axis=0)
    return data_resample, sr

def windows(data, window_size):
    start = 0
    while start < len(data):
        yield int(start), int(start + window_size)
        start += (window_size / 2)


# In[24]:


dataset = pd.read_csv('Multimodal_Databases/Customer_sentiment_Dataset/customer_sentiment_database.csv')
dataset['customer_sentiment'] = dataset['new_id'].astype(str) + '.wav'
dataset.head()


# In[25]:


dataset['emotion class'].value_counts()


# In[26]:


dataset['emotion class'].value_counts()


# In[27]:


from tqdm import tqdm
import librosa as lb
def feature_extractor_harmonic_percussive(path):
    data, simple_rate = lb.load(path)
    D = lb.stft(data)
    #melspec_harmonic, melspec_percussive = lb.decompose.hpss(D)
    #melspec_harmonic = np.mean(melspec_harmonic)
    #melspec_percussive = np.mean(melspec_percussive)
    #logspec_hp = np.average([melspec_harmonic, melspec_percussive])
    harmonic = lb.effects.harmonic(data,margin=5.0)
    percussive = librosa.effects.percussive(data, margin=5.0)
    #harmonic = np.mean(harmonic)
    #percussive = np.mean(percussive)
    logspec_hp = np.average([harmonic, percussive])
    return logspec_hp

def feature_extractor_logmelspectrogram(path):
    data, simple_rate = lb.load(path)
    data = lb.feature.melspectrogram(data)
    data = lb.power_to_db(data)
    data = np.mean(data,axis=1)
    return data



x_harmonic_percussive, x_logmelspectrogram, y = [], [], []
for path in tqdm(files):
    file_name = path.split('\\')[-1] #Return File Name
    class_label = path.split('-')[-1] #Return Class Id
    class_label = class_label.split('.')[0]
        
    x_harmonic_percussive.append(feature_extractor_harmonic_percussive(path))
    x_logmelspectrogram.append(feature_extractor_logmelspectrogram(path))
    y.append(class_label)

x_harmonic_percussive = np.array(x_harmonic_percussive)
x_logmelspectrogram = np.array(x_logmelspectrogram)
y = np.array(y)


# In[28]:


import re
emo_ids = []
for path in files:
    match = re.search(r'\d+', path)
    if match:
        number = int(match.group())
        emo_ids.append(number)


# In[29]:


from collections import Counter
y = dataset['emotion class'].values
Counter(y)


# In[30]:


final_df_emo = dataset.copy()


# In[31]:


x_harmonic_percussive.shape, x_logmelspectrogram.shape, y.shape


# In[32]:


features = np.hstack((x_harmonic_percussive.reshape(len(x_harmonic_percussive), 1), x_logmelspectrogram))
labels = y
features.shape, labels.shape


# In[33]:


feature_names = []
num_features = 129  # Number of features you want in the list

for i in range(1, num_features + 1):
    feature_name = f'emo_feature_{i}'
    feature_names.append(feature_name)

em_df = pd.DataFrame(np.hstack((x_harmonic_percussive.reshape(len(x_harmonic_percussive), 1), x_logmelspectrogram)), columns=feature_names)
em_df['new_id'] = emo_ids
em_df.head()


# In[34]:


emo_ids = dataset['new_id']
emo_ground_truths = dataset['emotion class']


# ### Train test Split

# In[35]:


from sklearn.metrics import cohen_kappa_score
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.30, shuffle=True)

print('Shape of Training Features : ', X_train.shape)
print('Shape of Testing Features : ', X_test.shape)

print('Shape of Training Labels : ', y_train.shape)
print('Shape of Testing Labels : ', y_test.shape)

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X_train, y_train = SMOTE(sampling_strategy='minority').fit_resample(X_train, y_train)
print('Train data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))


# In[36]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import learning_curve

# Define the XGBoost model
model = xgb.XGBClassifier()

# Define the hyperparameter grid
param_grid = {
    'max_depth': [7],
    'learning_rate': [0.1, 0.01, 0.001],
    'min_child_weight': [1, 3, 5]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[37]:


# Make predictions on the test data using the best model
# Make predictions on the test data using the best model
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]  # Use the predicted probabilities for the positive class

cm = confusion_matrix(y_test, predictions)

# Evaluate the best model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

# Print the evaluation metrics
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))


    
# Confusion matrix subplot
axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axs[0].set_title("Confusion Matrix", fontsize = 24)
tick_marks = np.arange(len(np.unique(y_test)))
axs[0].set_xticks(tick_marks)
axs[0].set_yticks(tick_marks)
axs[0].set_xticklabels(np.unique(y_test))
axs[0].set_yticklabels(np.unique(y_test))
axs[0].set_xlabel('Predicted Label')
axs[0].set_ylabel('True Label')
for i in range(len(np.unique(y_test))):
    for j in range(len(np.unique(y_test))):
        axs[0].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

        
# Learning curve subplot
train_sizes, train_scores, test_scores = learning_curve(best_model, X_train, y_train, cv=5, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 5))
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

# Learning curve subplot
axs[1].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
axs[1].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')
axs[1].set_xlabel('Training Set Size')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Learning Curve', fontsize = 24)
axs[1].legend(loc='lower right', facecolor='white')

# ROC curve subplot
axs[2].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
axs[2].plot([0, 1], [0, 1], color='red', linestyle='--')
axs[2].set_xlim([0.0, 1.0])
axs[2].set_ylim([0.0, 1.05])
axs[2].set_xlabel('False Positive Rate')
axs[2].set_ylabel('True Positive Rate')
axs[2].set_title('Receiver Operating Characteristic', fontsize = 24)
axs[2].legend(loc='lower right')

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('model_evaluation_EMODB_Model.png')
plt.suptitle('EMODB Model Performance', fontsize = 28, y = 1.1, color = 'red')
# Show the figure
plt.show();


# In[38]:


emo_model = best_model
emo_predictions = best_model.predict(features)
emo_modelled_data = pd.DataFrame({'new_id': emo_ids, 'emo_predictions':emo_predictions, 'ground_truths': labels})
emo_modelled_data.head()


# In[39]:


pd.crosstab(emo_modelled_data['emo_predictions'], emo_modelled_data['ground_truths'])


# # Churn Data

# In[40]:


observation_window1 = pd.read_csv('Multimodal_Databases/Churn Dataset/features_201806.csv')
observation_window2 = pd.read_csv('Multimodal_Databases/Churn Dataset/features_201812.csv')
observation_window3 = pd.read_csv('Multimodal_Databases/Churn Dataset/features_201906.csv')


# In[41]:


print('Shape of Observation Window 1 : ', observation_window1.shape)
print('Shape of Observation Window 2 : ', observation_window2.shape)
print('Shape of Outcome Window 1 : ', observation_window3.shape)


# In[42]:


features = pd.concat([observation_window1, observation_window2, observation_window3], axis=0, ignore_index=True)
features = features.drop_duplicates()
# features = features[features['acc_balance']>1500]
# features = features[features['acc_tenure']>6]
features = features.reset_index(drop = True)
#features.drop(['churn'], axis = 1, inplace = True)
print('Shape of Features : ', features.shape)
df = features.copy()
features.head()


# In[43]:


display(df['churn'].value_counts()/len(df)*100)
(df['churn'].value_counts()/len(df)*100).plot(kind = 'barh', figsize = (17,6));
plt.title('Distribution of Churn')
plt.show();


# ### Data Is Imbalanced

# In[44]:


print("Shape of Combined Dataframe : ", df.shape)


# ## Feature Engineering

# #### Dropping Duplicates

# In[45]:


finalDF = df.copy()
print("Shape of Combined Dataframe : ", finalDF.shape)


# In[46]:


finalDF.drop_duplicates(inplace=True) #subset='new_id', keep='last', inplace=True, ignore_index=False
print("Shape of Combined Dataframe : ", finalDF.shape)


# In[47]:


display(finalDF['churn'].value_counts()/len(finalDF)*100)
(finalDF['churn'].value_counts()/len(finalDF)*100).plot(kind = 'barh', figsize = (17,6));
plt.title('Distribution of Churn')
plt.show();


# In[48]:


finalDF.drop(['postcode'], axis = 1, inplace = True)
finalDF.drop(['group_no'], axis = 1, inplace = True)
finalDF.drop(['FL_new_id'], axis = 1, inplace = True)


# In[49]:


finalDF.head()


# In[50]:


finalDF['churn'] = finalDF['churn'].astype(object)
finalDF['age'] = finalDF['age'].astype(float)
finalDF['num_accounts'] = finalDF['num_accounts'].astype(float)
finalDF['acc_tenure'] = finalDF['acc_tenure'].astype(float)
finalDF['acc_balance'] = finalDF['acc_balance'].astype(float)
finalDF['num_options'] = finalDF['num_options'].astype(float)
finalDF['acc_balance_change_amount'] = finalDF['acc_balance_change_amount'].astype(float)
finalDF['num_options_change_amount'] = finalDF['num_options_change_amount'].astype(float)
finalDF['acc_balance_change_ratio'] = finalDF['acc_balance_change_ratio'].astype(float)
finalDF['num_accounts_change_ratio'] = finalDF['num_accounts_change_ratio'].astype(float)
finalDF['num_options_change_ratio'] = finalDF['num_options_change_ratio'].astype(float)
finalDF['fund_performance'] = finalDF['fund_performance'].astype(float)
finalDF['account_growth'] = finalDF['account_growth'].astype(float)
finalDF['account_growth_change'] = finalDF['account_growth_change'].astype(float)
finalDF['cust_tenure'] = finalDF['cust_tenure'].astype(float)
finalDF['dealer_change_recency'] = finalDF['dealer_change_recency'].astype(float)
finalDF['login_freq'] = finalDF['login_freq'].astype(float)
finalDF['login_recency'] = finalDF['login_recency'].astype(float)
finalDF['call_freq'] = finalDF['call_freq'].astype(float)
finalDF['call_recency'] = finalDF['call_recency'].astype(float)
finalDF['outflow_freq'] = finalDF['outflow_freq'].astype(float)
finalDF['outflow_recency'] = finalDF['outflow_recency'].astype(float)
finalDF['outflow_amount'] = finalDF['outflow_amount'].astype(float)
finalDF['outflow_ratio'] = finalDF['outflow_ratio'].astype(float)
finalDF['adviser_revenue_freq'] = finalDF['adviser_revenue_freq'].astype(float)
finalDF['adviser_revenue_amount'] = finalDF['adviser_revenue_amount'].astype(float)
finalDF['sg_amount'] = finalDF['sg_amount'].astype(float)
finalDF['salary_scr_amount'] = finalDF['salary_scr_amount'].astype(float)
finalDF['spouse_contr_amount'] = finalDF['spouse_contr_amount'].astype(float)
finalDF['personal_contr_amount'] = finalDF['personal_contr_amount'].astype(float)
finalDF['rollover_amount'] = finalDF['rollover_amount'].astype(float)
finalDF['contribution_amount'] = finalDF['contribution_amount'].astype(float)                                                  


# In[51]:


cols = ['churn', 'channel', 'gender', 'stmt_pref', 'annualrpt_pref', 'promotional_pref', 'returned_mail_count', 'rb_flag', 
'block_flag', 'has_email', 'has_work_tel', 'has_home_tel', 'has_mobile', 'postcode_change', 'adviser_no_change', 
'email_change', 'home_tel_change', 'work_tel_change', 'mobile_tel_change', 'sav_plan_change', 'dist_method_change', 
'fni_level_change', 'stmt_pref_change', 'annualrpt_pref_change', 'promotional_pref_change', 'option_changed', 
'num_accounts_closed', 'adviser_revenue_types', 'insurance_stopped', 'insurance_types', 'sg_freq', 'sg_stopped', 
'is_directed']

for col in cols:
    finalDF[col] = finalDF[col].astype(object)


# In[52]:


categoricals = finalDF.select_dtypes(include='object').columns.to_list()
summarize_categoricals(finalDF[categoricals], show_levels=True)


# In[53]:


numericals = list(set(finalDF.columns.tolist()) - set(categoricals)) + list(set(categoricals) - set(finalDF.columns.tolist()))


# In[54]:


labels = 'Non-Churn', 'Churn'
sizes = [finalDF.churn[finalDF['churn']==0].count(), finalDF.churn[finalDF['churn']==1].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show();


# ### Feature Engineering

# In[55]:


# Find columns with all same values
same_values_columns = finalDF.columns[finalDF.nunique() == 1]
same_values_columns


# In[56]:


# Delete columns with all same values
finalDF = finalDF.drop(columns=same_values_columns)


# In[57]:


corr = finalDF.corr()


# In[58]:



corr.style.background_gradient(axis=1)    .set_properties(**{'max-width': '80px', 'font-size': '10pt'})    .set_caption("Hover to magify")    .set_precision(2)    .set_table_styles(magnify())


# In[59]:


finalDF = remove_collinear_features(finalDF, threshold = 0.9)


# In[60]:


print('Dimensions of the Training set:',finalDF.shape)


# In[61]:


finalDF.head()


# In[62]:


final_df_churn = finalDF.copy()
# Extract categorical variables
categoricals = finalDF.select_dtypes(include='object').columns.to_list()
categoricals.remove('churn')
numeric_cols = list(final_df_churn.select_dtypes(include=np.number).columns)
numeric_cols.remove('new_id')


# In[63]:


# get the intial set of encoded features and encode them
final_df_churn = one_hot_encoding(final_df_churn, categoricals)
print('Dimensions of the Training set:',final_df_churn.shape)


# In[64]:


final_df_churn.head()


# In[65]:


numeric_cols = list(set(numeric_cols) - set(categoricals))


# In[66]:


### Splitting into features and labels dataframe
final_df_churn = final_df_churn.dropna()
final_df_churn.reset_index(inplace = True, drop = True)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
final_df_churn.loc[:,numeric_cols] = scaler.fit_transform(final_df_churn.loc[:,numeric_cols])

final_df_churn['churn'] = final_df_churn['churn'].astype('object')
final_df_churn.head()


# ## Late Fusion Churn

# In[67]:


finalDF = final_df_churn.copy()
churn_ids = final_df_churn['new_id']
finalDF['churn'] = finalDF['churn'].astype('object')
x = finalDF.drop(['churn', 'new_id', 'FL'], axis = 1)
y = finalDF['churn']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)


# #### Comparison Table of ML based Models - Train Test Split

# In[68]:


from sklearn.metrics import cohen_kappa_score
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, shuffle=True)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train = X_train.fillna(method='ffill')
X_test = X_test.fillna(method='ffill')


# In[69]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
Y_train = y_train.copy()


# In[70]:


print('Train data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))


# In[71]:


y_train.value_counts()


# In[72]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import learning_curve

# Define the XGBoost model
model = xgb.XGBClassifier()

# Define the hyperparameter grid
param_grid = {
    'max_depth': [7, 10, 15],
    'learning_rate': [0.1, 0.01, 0.001],
    'min_child_weight': [1, 3, 5]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[73]:


# Make predictions on the test data using the best model
# Make predictions on the test data using the best model
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]  # Use the predicted probabilities for the positive class

cm = confusion_matrix(y_test, predictions)

# Evaluate the best model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

# Print the evaluation metrics
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Confusion matrix subplot
axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axs[0].set_title("Confusion Matrix", fontsize = 24)
tick_marks = np.arange(len(np.unique(y_test)))
axs[0].set_xticks(tick_marks)
axs[0].set_yticks(tick_marks)
axs[0].set_xticklabels(np.unique(y_test))
axs[0].set_yticklabels(np.unique(y_test))
axs[0].set_xlabel('Predicted Label')
axs[0].set_ylabel('True Label')
for i in range(len(np.unique(y_test))):
    for j in range(len(np.unique(y_test))):
        axs[0].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

# Learning curve subplot
axs[1].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
axs[1].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')
axs[1].set_xlabel('Training Set Size')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Learning Curve', fontsize = 24)
axs[1].legend(loc='lower right', facecolor='white')

# ROC curve subplot
axs[2].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
axs[2].plot([0, 1], [0, 1], color='red', linestyle='--')
axs[2].set_xlim([0.0, 1.0])
axs[2].set_ylim([0.0, 1.05])
axs[2].set_xlabel('False Positive Rate')
axs[2].set_ylabel('True Positive Rate')
axs[2].set_title('Receiver Operating Characteristic', fontsize = 24)
axs[2].legend(loc='lower right')

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('model_evaluation_CHURN_Model_late_fusion.png')
plt.suptitle('CHURN Model Performance (Late Fusion)', fontsize = 28, y = 1.1, color = 'red')
# Show the figure
plt.show();


# In[74]:


churn_late_fusion_model = best_model
churn_late_fusion_predictions = best_model.predict(x)
churn_late_fusion_modelled_data = pd.DataFrame({'new_id': churn_ids, 'churn_late_fusion_predictions':churn_late_fusion_predictions, 'ground_truths': y})
churn_late_fusion_modelled_data.head()


# In[75]:


pd.crosstab(churn_late_fusion_modelled_data['churn_late_fusion_predictions'], churn_late_fusion_modelled_data['ground_truths'])


# ## Early Fusion Churn Dataset

# In[76]:


# Merge the datasets on the 'new_id' column
final_df_churn = final_df_churn.rename(columns={'FL': 'Financial Literacy'})
merged_df = pd.merge(final_df_churn, final_df_fl[['new_id', 'FL']], on='new_id')
merged_df = pd.merge(merged_df, final_df_emo[['new_id', 'emotion class']], on='new_id')
merged_df.head()


# In[77]:


finalDF = merged_df.copy()
churn_ids = finalDF['new_id']
finalDF['churn'] = finalDF['churn'].astype('object')
x = finalDF.drop(['churn', 'new_id'], axis = 1)
y = finalDF['churn']
categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)


# In[78]:


x.head()


# #### Comparison Table of ML based Models - Train Test Split

# In[79]:


from sklearn.metrics import cohen_kappa_score
y = y.astype(int)
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.20, shuffle=True)
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
X_train = X_train.fillna(method='ffill')
X_test = X_test.fillna(method='ffill')


# In[80]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

X_train, y_train = SMOTE().fit_resample(X_train, y_train)
Y_train = y_train.copy()


# In[81]:


print('Train data shape: {}'.format(X_train.shape))
print('Test data shape: {}'.format(X_test.shape))


# In[82]:


y_train.value_counts()


# In[83]:


import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import learning_curve

# Define the XGBoost model
model = xgb.XGBClassifier()

# Define the hyperparameter grid
param_grid = {
    'max_depth': [7, 10, 15],
    'learning_rate': [0.1, 0.01, 0.001],
    'min_child_weight': [1, 3, 5]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best hyperparameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_


# In[84]:


# Make predictions on the test data using the best model
predictions = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)[:, 1]  # Use the predicted probabilities for the positive class

# Evaluate the best model
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
roc_auc = roc_auc_score(y_test, probabilities)

# Print the evaluation metrics
print(f"Best Parameters: {best_params}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")


# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)
# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Confusion matrix subplot
axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
axs[0].set_title("Confusion Matrix", fontsize = 24)
tick_marks = np.arange(len(np.unique(y_test)))
axs[0].set_xticks(tick_marks)
axs[0].set_yticks(tick_marks)
axs[0].set_xticklabels(np.unique(y_test))
axs[0].set_yticklabels(np.unique(y_test))
axs[0].set_xlabel('Predicted Label')
axs[0].set_ylabel('True Label')
for i in range(len(np.unique(y_test))):
    for j in range(len(np.unique(y_test))):
        axs[0].text(j, i, cm[i, j], ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

# Learning curve subplot
axs[1].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
axs[1].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')
axs[1].set_xlabel('Training Set Size')
axs[1].set_ylabel('Accuracy')
axs[1].set_title('Learning Curve', fontsize = 24)
axs[1].legend(loc='lower right', facecolor='white')

# ROC curve subplot
axs[2].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.2f})')
axs[2].plot([0, 1], [0, 1], color='red', linestyle='--')
axs[2].set_xlim([0.0, 1.0])
axs[2].set_ylim([0.0, 1.05])
axs[2].set_xlabel('False Positive Rate')
axs[2].set_ylabel('True Positive Rate')
axs[2].set_title('Receiver Operating Characteristic', fontsize = 24)
axs[2].legend(loc='lower right')

# Adjust spacing between subplots
plt.tight_layout()

# Save the figure
plt.savefig('model_evaluation_CHURN_Model_early_fusion.png')
plt.suptitle('CHURN Model (EARLY FUSION) Model Performance', fontsize = 28, y = 1.1, color = 'red')
plt.show();


# In[85]:


x['FL'] = x['FL'].astype(int)


# In[86]:


churn_early_fusion_model = best_model
churn_early_fusion_predictions = best_model.predict(x)
churn_early_fusion_modelled_data = pd.DataFrame({'new_id': churn_ids, 'churn_early_fusion_predictions':churn_early_fusion_predictions, 'ground_truths': y})
churn_early_fusion_modelled_data.head()


# In[87]:


pd.crosstab(churn_early_fusion_modelled_data['churn_early_fusion_predictions'], churn_early_fusion_modelled_data['ground_truths'])


# # Multi model Ouput 

# In[88]:


# Merge the datasets on the 'new_id' column
final_df_churn = final_df_churn.rename(columns={'FL': 'Financial Literacy'})
merged_df = pd.merge(final_df_churn, final_df_fl[['new_id', 'FL']], on='new_id')
merged_df = pd.merge(merged_df, final_df_emo[['new_id', 'emotion class']], on='new_id')
merged_df.head()


# In[89]:


merged_df['multi_model_output'] = merged_df['churn'].apply(lambda x: 2 if x == 1 else 0)
merged_df.head()


# In[90]:


merged_df['multi_model_output'].value_counts()


# In[91]:


merged_df['FL'].value_counts()


# In[92]:


merged_df['emotion class'].value_counts()


# In[93]:


merged_df['multi_model_output'] = merged_df['multi_model_output'] + merged_df['FL'] + merged_df['emotion class']
merged_df.head()


# In[94]:


merged_df['multi_model_output'].value_counts()


# In[95]:


# Define the conditions and corresponding values
conditions = [
    merged_df['multi_model_output'] == 0,
    merged_df['multi_model_output'].isin([1, 2]),
    merged_df['multi_model_output'] >= 3
]
values = ['low-risk', 'mid-risk', 'high-risk']
# Apply the conditions using np.select()
merged_df['multi_model_output'] = np.select(conditions, values)


# In[96]:


merged_df['multi_model_output'].value_counts()


# In[97]:


# Define the conditions and corresponding values
conditions = [
    merged_df['multi_model_output'] == values[0],
    merged_df['multi_model_output'] == values[1],
    merged_df['multi_model_output'] == values[2]
]
# Apply the conditions using np.select()
merged_df['multi_model_output'] = np.select(conditions, [0, 1, 2])
merged_df['multi_model_output'].value_counts()


# ### Prediction of Low-Medium-High Risk Customer

# **Merged Datasets**
# 
# - There are two approaches
# 1. Late Fusion
# 2. Early Fusion
# 

# In[98]:


all_ids = merged_df['new_id']

merged_df.set_index('new_id', inplace = True)


# In[99]:


merged_df['multi_model_output'] = merged_df['multi_model_output'].astype(int)


# In[100]:


merged_df.head()


# In[101]:


late_fusion_data = merged_df.drop(['Financial Literacy', 'churn', 'FL', 'emotion class'], axis = 1)


# In[102]:


early_fusion_data = merged_df.drop(['Financial Literacy', 'churn', 'FL', 'emotion class'], axis = 1)


# # Late Fusion Modelling

# In[103]:


merged_df = pd.merge(late_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_late_fusion_modelled_data[['new_id', 'churn_late_fusion_predictions']], on='new_id')
late_fusion_data_ids = merged_df['new_id']
merged_df.head()

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)

all_features = features.copy()

features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_late_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[104]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the feature and label combinations
feature_combinations = {
    'all_features': features,
    'features_fc': features_fc,
    'features_fv': features_fv,
    'features_vc': features_vc
}

# Define the labels
labels = finalDF['multi_model_output']


# Define the models and their names
models = [
    SVC(kernel='linear', probability=True),
    RandomForestClassifier(random_state=2, max_depth=30, max_features='sqrt', n_estimators=50),
    XGBClassifier(random_state=2, max_depth=30, n_estimators=50)
]

model_names = [
    'Support Vector Machine',
    'Random Forest',
    'XGBoost'
]


result_tables = []


# In[105]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.context('seaborn-talk')
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': '16',
          'figure.figsize': (15, 5),
         'axes.labelsize': '20',
         'axes.titlesize':'30',
         'xtick.labelsize':'18',
         'ytick.labelsize':'18'}
plt.rcParams.update(params)

plt.rcParams['text.color'] = '#A04000'
plt.rcParams['xtick.color'] = '#283747'
plt.rcParams['ytick.color'] = '#808000'
plt.rcParams['axes.labelcolor'] = '#283747'

# Seaborn style (figures)
sns.set(context='notebook', style='whitegrid')
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[106]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.model_selection import KFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

# Iterate over each feature combination
for feature_name, feature_data in feature_combinations.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=2)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    result_dict = {
        'Model': [],
        'Fold': [],
        'Train Accuracy': [],
        'Test Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    # Iterate over each model
    for model, model_name in zip(models, model_names):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        fold = 1
        for train_index, val_index in kfold.split(feature_data):
            X_train_fold, X_val_fold = feature_data.iloc[train_index], feature_data.iloc[val_index]
            y_train_fold, y_val_fold = labels.iloc[train_index], labels.iloc[val_index]

            y_train_fold = label_encoder.fit_transform(y_train_fold)
            y_val_fold = label_encoder.transform(y_val_fold)

            model.fit(X_train_fold, y_train_fold)
            train_predictions = model.predict(X_train_fold)
            predictions = model.predict(X_val_fold)

            result_dict['Model'].append(model_name)
            result_dict['Fold'].append(fold)
            result_dict['Train Accuracy'].append(accuracy_score(y_train_fold, train_predictions))
            result_dict['Test Accuracy'].append(accuracy_score(y_val_fold, predictions))
            result_dict['Precision'].append(precision_score(y_val_fold, predictions, average='macro'))
            result_dict['Recall'].append(recall_score(y_val_fold, predictions, average='macro'))
            result_dict['F1 Score'].append(f1_score(y_val_fold, predictions, average='macro'))

            fold += 1

        result_table = pd.DataFrame(result_dict)
        result_table.set_index(['Model', 'Fold'], inplace=True)
        result_tables.append(result_table)

        # Turn off interactive mode
        plt.ioff()
        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[0].set_title(f"Confusion Matrix - {model_name} ({feature_name})", color='black')
        tick_marks = np.arange(len(label_encoder.classes_))
        axs[0].set_xticks(tick_marks)
        axs[0].set_yticks(tick_marks)
        axs[0].set_xticklabels(label_encoder.classes_, rotation=45)
        axs[0].set_yticklabels(label_encoder.classes_)
        axs[0].set_xlabel('Predicted Label')
        axs[0].set_ylabel('True Label')

        # Add text annotations for TP, FP, FN, TN
        for i in range(len(label_encoder.classes_)):
            for j in range(len(label_encoder.classes_)):
                axs[0].text(j, i, cm[i, j], ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')

        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, feature_data, labels, cv=5, scoring='accuracy', n_jobs=-1
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        axs[1].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
        axs[1].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')
        axs[1].set_xlabel('Training Set Size')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title(f'Learning Curve - {model_name} ({feature_name})', color='black')
        axs[1].legend(loc='lower right', facecolor='white')

        # ROC curve for each class
        if len(label_encoder.classes_) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, predictions)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            axs[2].plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
            axs[2].plot([0, 1], [0, 1], 'k--')
            axs[2].set_xlim([0.0, 1.0])
            axs[2].set_ylim([0.0, 1.05])
            axs[2].set_xlabel('False Positive Rate')
            axs[2].set_ylabel('True Positive Rate')
            axs[2].set_title(f"Receiver Operating Characteristic - {model_name} ({feature_name})", color='black')
            axs[2].legend(loc='lower right', facecolor='white')
        else:
            # Multiclass classification
            n_classes = len(label_encoder.classes_)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test, probabilities[:, i], pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot ROC curve for each class
            for i in range(n_classes):
                axs[2].plot(fpr[i], tpr[i], label='ROC curve (area = %0.4f)' % roc_auc[i])
            axs[2].plot([0, 1], [0, 1], 'k--')
            axs[2].set_xlim([0.0, 1.0])
            axs[2].set_ylim([0.0, 1.05])
            axs[2].set_xlabel('False Positive Rate')
            axs[2].set_ylabel('True Positive Rate')
            axs[2].set_title(f"Receiver Operating Characteristic - {model_name} ({feature_name})", color='black')
            axs[2].legend(loc='lower right', facecolor='white')

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3)
        plt.rcParams["text.color"] = "black"  # Set font color to black

        # Save the figures as separate image files
        confusion_matrix_filename = 'results/late/' + f'confusion_matrix_{model_name}_{feature_name}.png'
        learning_curve_filename = 'results/late/' + f'learning_curve_{model_name}_{feature_name}.png'
        roc_curve_filename = 'results/late/' + f'roc_curve_{model_name}_{feature_name}.png'

        plt.savefig(confusion_matrix_filename)
        plt.savefig(learning_curve_filename)
        plt.savefig(roc_curve_filename)

        # Show the plot
        plt.show();

plt.show();


# In[107]:


# Combine result tables into one DataFrame
combined_table = pd.concat(result_tables, keys=feature_combinations.keys())


# In[108]:


# Display the combined result table
display(combined_table)


# In[109]:


combined_table.to_csv('results_late_fusion.csv')


# # Early Fusion Modelling

# In[110]:


merged_df = pd.merge(early_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_early_fusion_modelled_data[['new_id', 'churn_early_fusion_predictions']], on='new_id')
merged_df.head()

early_fusion_data_ids = merged_df['new_id']

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)
all_features = features.copy()
features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_early_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[111]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the feature and label combinations
feature_combinations = {
    'all_features': features,
    'features_fc': features_fc,
    'features_fv': features_fv,
    'features_vc': features_vc
}

# Define the labels
labels = finalDF['multi_model_output']


# Define the models and their names
models = [
    SVC(kernel='linear', probability=True),
    RandomForestClassifier(random_state=2, max_depth=30, max_features='sqrt', n_estimators=50),
    XGBClassifier(random_state=2, max_depth=30, n_estimators=50)
]

model_names = [
    'Support Vector Machine',
    'Random Forest',
    'XGBoost'
]


result_tables = []


# In[112]:


import numpy as np
import matplotlib.pyplot as plt
plt.style.context('seaborn-talk')
plt.style.use('fivethirtyeight')
params = {'legend.fontsize': '16',
          'figure.figsize': (15, 5),
         'axes.labelsize': '20',
         'axes.titlesize':'30',
         'xtick.labelsize':'18',
         'ytick.labelsize':'18'}
plt.rcParams.update(params)

plt.rcParams['text.color'] = '#A04000'
plt.rcParams['xtick.color'] = '#283747'
plt.rcParams['ytick.color'] = '#808000'
plt.rcParams['axes.labelcolor'] = '#283747'

# Seaborn style (figures)
sns.set(context='notebook', style='whitegrid')
sns.set_style('ticks', {'xtick.direction':'in', 'ytick.direction':'in'})
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[113]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.model_selection import KFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

# Iterate over each feature combination
for feature_name, feature_data in feature_combinations.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=2)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    result_dict = {
        'Model': [],
        'Fold': [],
        'Train Accuracy': [],
        'Test Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    # Iterate over each model
    for model, model_name in zip(models, model_names):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        fold = 1
        for train_index, val_index in kfold.split(feature_data):
            X_train_fold, X_val_fold = feature_data.iloc[train_index], feature_data.iloc[val_index]
            y_train_fold, y_val_fold = labels.iloc[train_index], labels.iloc[val_index]

            y_train_fold = label_encoder.fit_transform(y_train_fold)
            y_val_fold = label_encoder.transform(y_val_fold)

            model.fit(X_train_fold, y_train_fold)
            train_predictions = model.predict(X_train_fold)
            predictions = model.predict(X_val_fold)

            result_dict['Model'].append(model_name)
            result_dict['Fold'].append(fold)
            result_dict['Train Accuracy'].append(accuracy_score(y_train_fold, train_predictions))
            result_dict['Test Accuracy'].append(accuracy_score(y_val_fold, predictions))
            result_dict['Precision'].append(precision_score(y_val_fold, predictions, average='macro'))
            result_dict['Recall'].append(recall_score(y_val_fold, predictions, average='macro'))
            result_dict['F1 Score'].append(f1_score(y_val_fold, predictions, average='macro'))

            fold += 1

        result_table = pd.DataFrame(result_dict)
        result_table.set_index(['Model', 'Fold'], inplace=True)
        result_tables.append(result_table)

        # Turn off interactive mode
        plt.ioff()
        # Create subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test)
        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        axs[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        axs[0].set_title(f"Confusion Matrix - {model_name} ({feature_name})", color='black')
        tick_marks = np.arange(len(label_encoder.classes_))
        axs[0].set_xticks(tick_marks)
        axs[0].set_yticks(tick_marks)
        axs[0].set_xticklabels(label_encoder.classes_, rotation=45)
        axs[0].set_yticklabels(label_encoder.classes_)
        axs[0].set_xlabel('Predicted Label')
        axs[0].set_ylabel('True Label')

        # Add text annotations for TP, FP, FN, TN
        for i in range(len(label_encoder.classes_)):
            for j in range(len(label_encoder.classes_)):
                axs[0].text(j, i, cm[i, j], ha='center', va='center',
                            color='white' if cm[i, j] > cm.max() / 2 else 'black')

        # Learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, feature_data, labels, cv=5, scoring='accuracy', n_jobs=-1
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        axs[1].plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training Accuracy')
        axs[1].plot(train_sizes, test_scores_mean, 'o-', color='g', label='Validation Accuracy')
        axs[1].set_xlabel('Training Set Size')
        axs[1].set_ylabel('Accuracy')
        axs[1].set_title(f'Learning Curve - {model_name} ({feature_name})', color='black')
        axs[1].legend(loc='lower right', facecolor='white')

        # ROC curve for each class
        if len(label_encoder.classes_) == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_test, predictions)
            roc_auc = auc(fpr, tpr)

            # Plot ROC curve
            axs[2].plot(fpr, tpr, label='ROC curve (area = %0.4f)' % roc_auc)
            axs[2].plot([0, 1], [0, 1], 'k--')
            axs[2].set_xlim([0.0, 1.0])
            axs[2].set_ylim([0.0, 1.05])
            axs[2].set_xlabel('False Positive Rate')
            axs[2].set_ylabel('True Positive Rate')
            axs[2].set_title(f"Receiver Operating Characteristic - {model_name} ({feature_name})", color='black')
            axs[2].legend(loc='lower right', facecolor='white')
        else:
            # Multiclass classification
            n_classes = len(label_encoder.classes_)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test, probabilities[:, i], pos_label=i)
                roc_auc[i] = auc(fpr[i], tpr[i])

            # Plot ROC curve for each class
            for i in range(n_classes):
                axs[2].plot(fpr[i], tpr[i], label='ROC curve (area = %0.4f)' % roc_auc[i])
            axs[2].plot([0, 1], [0, 1], 'k--')
            axs[2].set_xlim([0.0, 1.0])
            axs[2].set_ylim([0.0, 1.05])
            axs[2].set_xlabel('False Positive Rate')
            axs[2].set_ylabel('True Positive Rate')
            axs[2].set_title(f"Receiver Operating Characteristic - {model_name} ({feature_name})", color='black')
            axs[2].legend(loc='lower right', facecolor='white')

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0.3)
        plt.rcParams["text.color"] = "black"  # Set font color to black

        # Save the figures as separate image files
        confusion_matrix_filename = 'results/early/' + f'confusion_matrix_{model_name}_{feature_name}.png'
        learning_curve_filename = 'results/early/' + f'learning_curve_{model_name}_{feature_name}.png'
        roc_curve_filename = 'results/early/' + f'roc_curve_{model_name}_{feature_name}.png'

        plt.savefig(confusion_matrix_filename)
        plt.savefig(learning_curve_filename)
        plt.savefig(roc_curve_filename)

        # Show the plot
        plt.show();

plt.show();


# In[114]:


# Combine result tables into one DataFrame
combined_table = pd.concat(result_tables, keys=feature_combinations.keys())


# In[115]:


# Display the combined result table
display(combined_table)


# In[116]:


combined_table.to_csv('results_early_fusion.csv')


# ## Correlation Analysis

# In[117]:


# Merge final_df_emo and final_df_fl on 'new_id'
merged_df = pd.merge(final_df_emo, final_df_fl, on='new_id', how='inner')
# Merge merged_df and final_df_churn on 'new_id'
final_merged_df = pd.merge(merged_df, final_df_churn, on='new_id', how='inner')


# In[118]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Select the desired categorical features and the target variable
selected_features = ['emotion class', 'FL', 'churn']
final_merged_df = final_merged_df[selected_features].copy()

# Convert categorical variables to numeric labels
label_encoder = LabelEncoder()
for feature in selected_features:
    final_merged_df[feature] = label_encoder.fit_transform(final_merged_df[feature])

# Calculate the correlation matrix
correlation_matrix = final_merged_df.corr()

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Analysis - Categorical Features and Multimodal Performance')
plt.show();


# # Feature Importance vs. Performance

# ### Late Fusion

# In[119]:


merged_df = pd.merge(late_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_late_fusion_modelled_data[['new_id', 'churn_late_fusion_predictions']], on='new_id')
late_fusion_data_ids = merged_df['new_id']
merged_df.head()

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)

all_features = features.copy()

features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_late_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[120]:


get_ipython().run_line_magic('matplotlib', 'inline')

# Create empty lists to store the results
num_features = list(range(1, 143, 10))
accuracy_scores = []
feature_combinations = {
    'all_features': all_features,
    'features_fc': features_fc,
    'features_fv': features_fv,
    'features_vc': features_vc
}

# Iterate over the number of features
for num in num_features:
    accuracy_scores_combination = []  # Store accuracy scores for each feature combination
    
    # Iterate over each feature combination
    for feature_name, feature_data in feature_combinations.items():
        X = feature_data.iloc[:, :num]  # Select the first 'num' features
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=2)

        model = RandomForestClassifier(max_depth = 15)

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores_combination.append(accuracy)  # Store the accuracy score for the current feature combination
    
    # Store the accuracy scores for the current number of features
    accuracy_scores.append(accuracy_scores_combination)


# In[121]:


# Create a fancier plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each feature combination
colors = ['blue', 'green', 'red', 'orange']
# Plot the lines with markers
for i, feature_name in enumerate(feature_combinations.keys()):
    plt.plot(num_features, [score[i] for score in accuracy_scores], marker='o', label=feature_name, color=colors[i], linewidth=2)

# Set axis labels and title
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.title('Number of Features vs. Accuracy Score (Late Fusion Model)', fontsize=14)

# Customize tick labels and grid
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5)

# Add legend with a fancy box
legend = plt.legend(loc='lower right', fontsize=10, frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')

# Adjust layout and spacing
plt.tight_layout()
# Save the plot if needed
plt.savefig('Number of Features vs. Accuracy Score-Late Fusion Model.png', dpi=300)

# Show the plot
plt.show();


# ## Early Fusion

# In[122]:


merged_df = pd.merge(early_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_early_fusion_modelled_data[['new_id', 'churn_early_fusion_predictions']], on='new_id')
merged_df.head()

early_fusion_data_ids = merged_df['new_id']

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)

all_features = features.copy()

features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_early_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[123]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Create empty lists to store the results
num_features = list(range(1, 143, 10))
accuracy_scores = []
feature_combinations = {
    'all_features': all_features,
    'features_fc': features_fc,
    'features_fv': features_fv,
    'features_vc': features_vc
}

# Iterate over the number of features
for num in num_features:
    accuracy_scores_combination = []  # Store accuracy scores for each feature combination
    
    # Iterate over each feature combination
    for feature_name, feature_data in feature_combinations.items():
        X = feature_data.iloc[:, :num]
        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=2)

        model = RandomForestClassifier(max_depth=15)

        # Fit the model
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate the accuracy score
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_scores_combination.append(accuracy)  # Store the accuracy score for the current feature combination
    
    # Store the accuracy scores for the current number of features
    accuracy_scores.append(accuracy_scores_combination)


# In[124]:


# Create a fancier plot
fig, ax = plt.subplots(figsize=(10, 6))

# Define colors for each feature combination
colors = ['blue', 'green', 'red', 'orange']
# Plot the lines with markers
for i, feature_name in enumerate(feature_combinations.keys()):
    plt.plot(num_features, [score[i] for score in accuracy_scores], marker='o', label=feature_name, color=colors[i], linewidth=2)

# Set axis labels and title
plt.xlabel('Number of Features', fontsize=12)
plt.ylabel('Accuracy Score', fontsize=12)
plt.title('Number of Features vs. Accuracy Score (Early Fusion Model)', fontsize=14)

# Customize tick labels and grid
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, linestyle='--', linewidth=0.5)

# Add legend with a fancy box
legend = plt.legend(loc='lower right', fontsize=10, frameon=True)
frame = legend.get_frame()
frame.set_facecolor('white')
frame.set_edgecolor('black')

# Adjust layout and spacing
plt.tight_layout()

# Save the plot if needed
plt.savefig('Number of Features vs. Accuracy Score-Early Fusion Model.png', dpi=300)

# Show the plot
plt.show();


# ## Risk Distribution
# Compare the number of Low-Risk, Mid-Risk, and High-Risk into a plot for three below models

# In[125]:


merged_df = pd.merge(final_df_churn, final_df_fl, on='new_id')
merged_df['multi_model_output'] = merged_df['churn'].apply(lambda x: 2 if x == 1 else 0)
merged_df = pd.merge(merged_df, final_df_emo[['new_id', 'emotion class']], on='new_id')

merged_df['multi_model_output'] = merged_df['multi_model_output'] + merged_df['FL'] + merged_df['emotion class']
merged_df.head()


# In[126]:


all_features_copy = merged_df.copy() 
all_features_copy.head()


# In[127]:


churn_features = all_features_copy.drop(['new_id', 'churn', 'Financial Literacy', 'FL', 
                                    'Change_12m_Cash', 'Change_12m_Fixed_interest', 'Change_12m_Global_fixed_interest', 
                                    'Change_12m_Australian_share', 'Change_12m_Global_share', 'Change_12m_Property', 
                                    'FNILogin', 'Investor_Inbound', 'ONLINE-Switches', 'Personal_Non_conc', 'salary_Sacr', 
                                    'super_guarantee', 'Switch', 'Withdrawal', 'personal_non_conc_amt', 'personal_conc_amt', 
                                    'salary_sacr_amt', 'Gov_co_cont_amt', 'SG_amt', 'Total_concessional', 'Total_non_concessional', 
                                    'concessional_cap', 'Investor_Outbound_0', 'Investor_Outbound_1', 'Investor_Outbound_2', 
                                    'Investor_Outbound_3', 'Investor_Call Transfer_0', 'Investor_Call Transfer_1', 
                                    'Investor_Call Transfer_2', 'Investor_Call Transfer_3', 'Personal_conc_0', 'Personal_conc_1',
                                    'Personal_conc_2', 'Personal_conc_3', 'Personal_conc_7', 'Personal_conc_12', 'Personal_conc_18', 
                                    'ADDRESS_CHANGE_0', 'ADDRESS_CHANGE_1', 'Rollover_0', 'Rollover_1', 'Rollover_2', 'Rollover_4', 
                                    'Investor_Email_0', 'Investor_Email_1', 'Investor_Email_2', 'Rollover_wdl_0', 'Rollover_wdl_1', 
                                    'Rollover_wdl_2', 'Non-member_Outbound_0', 'Non-member_Outbound_1', 'Non-member_Outbound_2', 
                                    'Non-member_Outbound_3', 'multi_model_output', 'emotion class'], axis = 1)


# In[128]:


all_features_copy = pd.merge(all_features_copy, em_df, on = 'new_id', how = 'inner')


# In[129]:


all_features_copy.head()


# In[130]:


churn_columns = final_df_churn.columns.tolist()
churn_columns.extend(['new_id', 'Financial Literacy', 'churn', 'FL', 'multi_model_output', 'emotion class'])
fl_features = all_features_copy.drop(churn_columns, axis=1)
em_features = all_features_copy[feature_names]


# In[131]:


churn_features.shape, fl_features.shape, em_features.shape, labels.shape


# In[132]:


# Define the feature and label combinations
model_combinations = {
    'Churn Model': churn_features,
    'FL Model': fl_features,
    'Emotion DB Model': em_features
}

# Define the conditions and corresponding values
conditions = [
    merged_df['multi_model_output'] == 0,
    merged_df['multi_model_output'].isin([1, 2]),
    merged_df['multi_model_output'] >= 3
]
values = ['low-risk', 'mid-risk', 'high-risk']
# Apply the conditions using np.select()
all_features_copy['multi_model_output'] = np.select(conditions, values)

# Define the labels
labels = all_features_copy['multi_model_output']

# Define the models and their names
models = [
    RandomForestClassifier(random_state=2, max_depth=30, max_features='sqrt', n_estimators=50)
]

model_names = [
    'Random Forest']

preds = []
result_tables = []


# In[133]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Iterate over each feature combination
for feature_name, feature_data in model_combinations.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=2)

    #label_encoder = LabelEncoder()
    #y_train = label_encoder.fit_transform(y_train)
    #y_test = label_encoder.transform(y_test)

    result_dict = {
        'Model': [],
        'Train Accuracy': [],
        'Test Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }

    # Iterate over each model
    for model, model_name in zip(models, model_names):
        model.fit(X_train, y_train)
        train_predictions = model.predict(X_train)
        predictions = model.predict(X_test)
        all_predictions = model.predict(feature_data)
        print('='*len(feature_name))
        print(feature_name)
        print('='*len(feature_name))
        print('Number of Low-Risk, Mid-Risk, and High-Risk')
        print(Counter(all_predictions))
        
        from collections import Counter

        # Count the occurrences of each prediction
        prediction_counts = Counter(all_predictions)
        preds.append(prediction_counts)


        # Get the labels and counts for the bar plot
        labels_preds = prediction_counts.keys()
        counts = prediction_counts.values()

        # Create the bar plot
        plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
        bars = plt.bar(labels_preds, counts)

        # Add data labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, height,
                     ha='center', va='bottom', fontsize=12)

        plt.title('Prediction Distribution for ' + feature_name, fontsize=16)
        plt.xlabel('Prediction', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.xticks(rotation=45, fontsize=10)
        plt.yticks(fontsize=10)
        plt.grid(axis='y', linestyle='--')  # Add a dashed grid line on the y-axis

        # Add a background color for better visibility
        plt.gca().set_facecolor('#f5f5f5')

        # Show the plot
        plt.show();

        result_dict['Model'].append(model_name)
        result_dict['Train Accuracy'].append(accuracy_score(y_train, train_predictions))
        result_dict['Test Accuracy'].append(accuracy_score(y_test, predictions))
        result_dict['Precision'].append(precision_score(y_test, predictions, average='macro'))
        result_dict['Recall'].append(recall_score(y_test, predictions, average='macro'))
        result_dict['F1 Score'].append(f1_score(y_test, predictions, average='macro'))

        result_table = pd.DataFrame(result_dict)
        result_table.set_index(['Model'], inplace=True)
        result_tables.append(result_table)

        
plt.show();        


# ### Late Fusion

# In[134]:


late_fusion_data = late_fusion_data.reset_index()
merged_df = pd.merge(late_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id', how='inner')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id', how='inner')
merged_df['new_id'] = merged_df['new_id'].astype(int)
churn_late_fusion_modelled_data['new_id'] = churn_late_fusion_modelled_data['new_id'].astype(int)
churn_late_fusion_modelled_data = churn_late_fusion_modelled_data.drop_duplicates(subset='new_id', keep='first')
merged_df = pd.merge(merged_df, churn_late_fusion_modelled_data[['new_id', 'churn_late_fusion_predictions']], on='new_id', how='inner')


# In[135]:


finalDF = merged_df.copy()
finalDF['multi_model_output'] = np.select(conditions, values)

features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)
predictions = model.predict(X_test)
all_predictions = model.predict(features)
feature_name = 'Late Fusion Model'
print('='*len(feature_name))
print(feature_name)
print('='*len(feature_name))
print('Number of Low-Risk, Mid-Risk, and High-Risk')
print(Counter(all_predictions))

# Count the occurrences of each prediction
prediction_counts = Counter(all_predictions)
late_prediction_counts = Counter(all_predictions)

# Get the labels and counts for the bar plot
labels_preds = prediction_counts.keys()
counts = prediction_counts.values()

# Create the bar plot
plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
bars = plt.bar(labels_preds, counts)

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, height,
             ha='center', va='bottom', fontsize=12)

plt.title('Prediction Distribution for ' + feature_name, fontsize=16)
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--')  # Add a dashed grid line on the y-axis

# Add a background color for better visibility
plt.gca().set_facecolor('#f5f5f5')

# Show the plot
plt.show();


# ### Early Fusion

# In[136]:


early_fusion_data = early_fusion_data.reset_index()
merged_df = pd.merge(early_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id', how='inner')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id', how='inner')
merged_df['new_id'] = merged_df['new_id'].astype(int)
churn_early_fusion_modelled_data['new_id'] = churn_early_fusion_modelled_data['new_id'].astype(int)
churn_early_fusion_modelled_data = churn_early_fusion_modelled_data.drop_duplicates(subset='new_id', keep='first')
merged_df = pd.merge(merged_df, churn_early_fusion_modelled_data[['new_id', 'churn_early_fusion_predictions']], on='new_id', how='inner')


# In[137]:


finalDF = merged_df.copy()
finalDF['multi_model_output'] = np.select(conditions, values)
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=2)

model = RandomForestClassifier()
model.fit(X_train, y_train)
train_predictions = model.predict(X_train)
predictions = model.predict(X_test)
all_predictions = model.predict(features)
feature_name = 'Early Fusion Model'
print('='*len(feature_name))
print(feature_name)
print('='*len(feature_name))
print('Number of Low-Risk, Mid-Risk, and High-Risk')
print(Counter(all_predictions))

# Count the occurrences of each prediction
prediction_counts = Counter(all_predictions)
early_prediction_counts = Counter(all_predictions)
# Get the labels and counts for the bar plot
labels_preds = prediction_counts.keys()
counts = prediction_counts.values()

# Create the bar plot
plt.figure(figsize=(10, 4))  # Adjust the figure size as needed
bars = plt.bar(labels_preds, counts)

# Add data labels on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, height,
             ha='center', va='bottom', fontsize=12)

plt.title('Prediction Distribution for ' + feature_name, fontsize=16)
plt.xlabel('Prediction', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--')  # Add a dashed grid line on the y-axis

# Add a background color for better visibility
plt.gca().set_facecolor('#f5f5f5')

# Show the plot
plt.show();


# In[138]:


preds


# In[139]:


late_prediction_counts


# In[140]:


early_prediction_counts


# In[141]:


resultsdf = pd.DataFrame([preds[0], preds[1], preds[2], late_prediction_counts, early_prediction_counts])
resultsdf.index = ['Baseline Model', 'Financial Literacy Model', 'Emotion DB Model', 'Late Fusion Model', 'Early Fusion Model']
resultsdf


# In[142]:


ax = resultsdf.plot(kind='bar', rot=0, figsize=(10, 6), width=0.7)

plt.xlabel('Predictions')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Predictions')

# Annotate the bars with their respective counts
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show();


# In[143]:


ax = resultsdf.T.plot(kind='bar', rot=0, figsize=(10, 6), width=0.7)

plt.xlabel('Predictions')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Predictions')

# Annotate the bars with their respective counts
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show();


# In[144]:


resultsdf = pd.DataFrame([preds[0], late_prediction_counts, early_prediction_counts])
resultsdf.index = ['Baseline Model', 'Late Fusion Model', 'Early Fusion Model']
resultsdf


# In[145]:


ax = resultsdf.plot(kind='bar', rot=0, figsize=(10, 6), width=0.7)

plt.xlabel('Predictions')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Predictions')

# Annotate the bars with their respective counts
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show();


# In[146]:


ax = resultsdf.T.plot(kind='bar', rot=0, figsize=(10, 6), width=0.7)

plt.xlabel('Predictions')
plt.ylabel('Count')
plt.title('Grouped Bar Chart of Predictions')

# Annotate the bars with their respective counts
for p in ax.patches:
    ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width() / 2, p.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.legend(loc='upper right')
plt.tight_layout()
plt.show();


# 
# 
# ### Macro Averaged F1 Score

# # Late Fusion Modelling

# In[147]:


merged_df = pd.merge(late_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_late_fusion_modelled_data[['new_id', 'churn_late_fusion_predictions']], on='new_id')
late_fusion_data_ids = merged_df['new_id']

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)

all_features = features.copy()

features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_late_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[148]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the feature and label combinations
feature_combinations = {
    'all_features': features,
    'features_fc': features_fc,
    'features_fv': features_fv,
    'features_vc': features_vc
}

# Define the labels
labels = finalDF['multi_model_output']


# Define the models and their names
models = [
    RandomForestClassifier(random_state=2, max_depth=30, max_features='sqrt', n_estimators=50)]

model_names = [
    'Random Forest'
]

result_tables = []


# In[149]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.model_selection import KFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

# Iterate over each feature combination
for feature_name, feature_data in feature_combinations.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=2)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    result_dict = {
        'Model': [],
        'Fold': [],
        'Macro Average F1 Score': []
    }

    # Iterate over each model
    for model, model_name in zip(models, model_names):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        fold = 1
        for train_index, val_index in kfold.split(feature_data):
            X_train_fold, X_val_fold = feature_data.iloc[train_index], feature_data.iloc[val_index]
            y_train_fold, y_val_fold = labels.iloc[train_index], labels.iloc[val_index]

            y_train_fold = label_encoder.fit_transform(y_train_fold)
            y_val_fold = label_encoder.transform(y_val_fold)

            model.fit(X_train_fold, y_train_fold)
            train_predictions = model.predict(X_train_fold)
            predictions = model.predict(X_val_fold)

            result_dict['Model'].append(model_name)
            result_dict['Fold'].append(fold)
            result_dict['Macro Average F1 Score'].append(f1_score(y_val_fold, predictions, average='macro'))

            fold += 1

        result_table = pd.DataFrame(result_dict)
        result_table.set_index(['Model', 'Fold'], inplace=True)
        result_tables.append(result_table)


# In[150]:


# Combine result tables into one DataFrame
combined_table = pd.concat(result_tables, keys=feature_combinations.keys())


# In[151]:


# Display the combined result table
display(combined_table)


# # Early Fusion Modelling

# In[152]:


merged_df = pd.merge(early_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_early_fusion_modelled_data[['new_id', 'churn_early_fusion_predictions']], on='new_id')
merged_df.head()

early_fusion_data_ids = merged_df['new_id']

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)
all_features = features.copy()
features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_early_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[153]:


get_ipython().run_line_magic('matplotlib', 'inline')
result_tables = []

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.model_selection import KFold, learning_curve, train_test_split
from sklearn.preprocessing import LabelEncoder

# Iterate over each feature combination
for feature_name, feature_data in feature_combinations.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=2)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    result_dict = {
        'Model': [],
        'Fold': [],
        'Macro Average F1 Score': []
    }

    # Iterate over each model
    for model, model_name in zip(models, model_names):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        fold = 1
        for train_index, val_index in kfold.split(feature_data):
            X_train_fold, X_val_fold = feature_data.iloc[train_index], feature_data.iloc[val_index]
            y_train_fold, y_val_fold = labels.iloc[train_index], labels.iloc[val_index]

            y_train_fold = label_encoder.fit_transform(y_train_fold)
            y_val_fold = label_encoder.transform(y_val_fold)

            model.fit(X_train_fold, y_train_fold)
            train_predictions = model.predict(X_train_fold)
            predictions = model.predict(X_val_fold)

            result_dict['Model'].append(model_name)
            result_dict['Fold'].append(fold)
            result_dict['Macro Average F1 Score'].append(f1_score(y_val_fold, predictions, average='macro'))

            fold += 1

        result_table = pd.DataFrame(result_dict)
        result_table.set_index(['Model', 'Fold'], inplace=True)
        result_tables.append(result_table)


# In[154]:


# Combine result tables into one DataFrame
combined_table = pd.concat(result_tables, keys=feature_combinations.keys())


# In[155]:


# Display the combined result table
display(combined_table)


# In[156]:


combined_table.to_csv('results_early_fusion-f1.csv')


# ##  Mean Average Precision Score (MAP)

# # Late Fusion Modelling

# In[176]:


merged_df = pd.merge(late_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_late_fusion_modelled_data[['new_id', 'churn_late_fusion_predictions']], on='new_id')
late_fusion_data_ids = merged_df['new_id']

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)

all_features = features.copy()

features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_late_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[177]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the feature and label combinations
feature_combinations = {
    'all_features': features,
    'features_fc': features_fc,
    'features_fv': features_fv,
    'features_vc': features_vc
}

# Define the labels
labels = finalDF['multi_model_output']


# Define the models and their names
models = [
    RandomForestClassifier(random_state=2, max_depth=30, max_features='sqrt', n_estimators=50)]

model_names = [
    'Random Forest'
]

result_tables = []


# In[181]:


get_ipython().run_line_magic('matplotlib', 'inline')
result_tables = []

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Iterate over each feature combination
for feature_name, feature_data in feature_combinations.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.2, random_state=2)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    result_dict = {
        'Model': [],
        'Fold': [],
        'Mean Average Precision Score': []
    }

    # Iterate over each model
    for model, model_name in zip(models, model_names):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        fold = 1
        for train_index, val_index in kfold.split(feature_data):
            X_train_fold, X_val_fold = feature_data.iloc[train_index], feature_data.iloc[val_index]
            y_train_fold, y_val_fold = labels.iloc[train_index], labels.iloc[val_index]

            y_train_fold = label_encoder.fit_transform(y_train_fold)
            y_val_fold = label_encoder.transform(y_val_fold)

            model.fit(X_train_fold, y_train_fold)
            train_predictions = model.predict(X_train_fold)
            predictions = model.predict(X_val_fold)

            # Convert to one-hot encoding
            one_hot_predictions = label_encoder.transform(predictions)
            one_hot_y_val_fold = label_encoder.transform(y_val_fold)

            average_precision = average_precision_score(
                pd.get_dummies(one_hot_y_val_fold),
                pd.get_dummies(one_hot_predictions),
                average="macro"
            )

            result_dict['Model'].append(model_name)
            result_dict['Fold'].append(fold)
            result_dict['Mean Average Precision Score'].append(average_precision)

            fold += 1

        result_table = pd.DataFrame(result_dict)
        result_table.set_index(['Model', 'Fold'], inplace=True)
        result_tables.append(result_table)


# In[182]:


# Combine result tables into one DataFrame
combined_table = pd.concat(result_tables, keys=feature_combinations.keys())


# In[183]:


# Display the combined result table
display(combined_table)


# # Early Fusion Modelling

# In[190]:


merged_df = pd.merge(early_fusion_data, fl_modelled_data[['new_id', 'fl_predictions']], on='new_id')
merged_df = pd.merge(merged_df, emo_modelled_data[['new_id', 'emo_predictions']], on='new_id')
merged_df = pd.merge(merged_df, churn_early_fusion_modelled_data[['new_id', 'churn_early_fusion_predictions']], on='new_id')
merged_df.head()

early_fusion_data_ids = merged_df['new_id']

finalDF = merged_df.copy()
features = finalDF.drop(['new_id', 'multi_model_output'], axis = 1)
labels = finalDF['multi_model_output']

categorical_columns = list(x.select_dtypes(include='category').columns)
numeric_columns = list(x.select_dtypes(exclude='category').columns)
all_features = features.copy()
features_fc = features.drop(['emo_predictions'], axis = 1)
features_fv = features[['churn_early_fusion_predictions']]
features_vc = features.drop(['fl_predictions'], axis = 1)
labels = finalDF['multi_model_output']
labels = labels.astype(int)


# In[195]:


get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the feature and label combinations
feature_combinations = {
    'all_features': features,
    'features_fc': features_fc,
    'features_fv': features_fv,
    'features_vc': features_vc
}


# Define the models and their names
models = [
    RandomForestClassifier(random_state=2, max_depth=30, max_features='sqrt', n_estimators=50)]

model_names = [
    'Random Forest'
]

result_tables = []


# In[196]:


get_ipython().run_line_magic('matplotlib', 'inline')
result_tables = []

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve
)
from sklearn.metrics import average_precision_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier

# Iterate over each feature combination
for feature_name, feature_data in feature_combinations.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.3, random_state=2)
    
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    result_dict = {
        'Model': [],
        'Fold': [],
        'Mean Average Precision Score': []
    }

    # Iterate over each model
    for model, model_name in zip(models, model_names):
        kfold = KFold(n_splits=5, shuffle=True, random_state=2)
        fold = 1
        for train_index, val_index in kfold.split(feature_data):
            X_train_fold, X_val_fold = feature_data.iloc[train_index], feature_data.iloc[val_index]
            y_train_fold, y_val_fold = labels.iloc[train_index], labels.iloc[val_index]

            y_train_fold = label_encoder.fit_transform(y_train_fold)
            y_val_fold = label_encoder.transform(y_val_fold)

            model.fit(X_train_fold, y_train_fold)
            train_predictions = model.predict(X_train_fold)
            predictions = model.predict(X_val_fold)

            # Convert to one-hot encoding
            one_hot_predictions = label_encoder.transform(predictions)
            one_hot_y_val_fold = label_encoder.transform(y_val_fold)

            average_precision = average_precision_score(
                pd.get_dummies(one_hot_y_val_fold),
                pd.get_dummies(one_hot_predictions),
                average="macro"
            )

            result_dict['Model'].append(model_name)
            result_dict['Fold'].append(fold)
            result_dict['Mean Average Precision Score'].append(average_precision)

            fold += 1

        result_table = pd.DataFrame(result_dict)
        result_table.set_index(['Model', 'Fold'], inplace=True)
        result_tables.append(result_table)


# In[197]:


# Combine result tables into one DataFrame
combined_table = pd.concat(result_tables, keys=feature_combinations.keys())


# In[198]:


# Display the combined result table
display(combined_table)


# In[199]:


combined_table.to_csv('results_early_fusion-f1.csv')


# In[ ]:





# In[ ]:




