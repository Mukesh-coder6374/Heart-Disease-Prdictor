import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import joblib
import matplotlib.pyplot as plt

#1.Importing the dataset
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/content/drive/MyDrive/heart_disease_uci.csv')

#2.Datapreprocessing
np.set_printoptions(precision=1, floatmode='fixed')  # Always show one decimal
np.set_printoptions(suppress=True)
##2.1 Removing the unwanted features(metadata)
df = df.drop(columns = ['id','dataset']).rename(columns = {'num':'target'})
##2.2 Scaling the target from (0 to 4) to (0 to 1)
df['target'] = df['target'].apply(lambda x:1 if x>0 else 0)
#2.3 Coverting Boolean
df['fbs'] = df['fbs'].map({'True':1,'False':0})
df['exang']=df['exang'].map({'True':1,'False':0})
#2.4 Encoding categorical data
categorical_col = ['cp','restecg','slope','thal']
df = pd.get_dummies(df,columns = categorical_col,drop_first = True)
bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)
col_to_move = df.pop('target')  # Removes 'colA' and returns it as a Series
df['target'] = col_to_move     # Adds 'colA' back at the end
df['sex'] = df['sex'].map({'Male': 1, 'Female': 0})
df = df.drop(columns=['fbs', 'exang'])
numerical_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
##2.5 Handelling missing data
imputer = SimpleImputer(strategy='median')
df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

#3 Splitting the dataset
##3.1 features matrix (x) and target vectors(y)
x = df.iloc[:,:-1]
y = df.iloc[:,-1]
##3.2 Splitting train(80%) and test(20%)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 67)
##3.3 training the model(RandomForestClassifier)
model = RandomForestClassifier(random_state=67)
model.fit(x_train,y_train)
##3.4 preducting test features
y_pred = model.predict(x_test)
##3.5 evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4 SHAP
import shap
shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values,x_test)


# Create a SHAP summary bar plot (feature importance)
plt.figure()
shap.summary_plot(shap_values, x_test, plot_type="bar", show=False)

# Save the plot as PNG
plt.savefig("/content/drive/MyDrive/shap_feature_importance.png", bbox_inches='tight')
import joblib

# Save the model
joblib.dump(model, 'heart_disease_model.pkl')
from google.colab import files
files.download('heart_disease_model.pkl')