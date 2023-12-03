import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
import numpy as np

# Step 1: Load and preprocess the data
data = pd.read_csv('train.csv')
data.dropna(inplace=True)
label_encoder = LabelEncoder()
data['Payment_Behaviour'] = label_encoder.fit_transform(data['Payment_Behaviour'])
features = data[['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Payment_Behaviour']]
target = data['Credit_Score']

# Step 2: Perform Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 8))
sns.pairplot(data, hue='Credit_Score', diag_kind='kde')
plt.show()

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train the models
# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Gradient Boosting
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

# Support Vector Machine (SVM)
svm_model = SVC(probability=True)
svm_model.fit(X_train, y_train)

# Step 5: Evaluate the models
models = [logistic_model, rf_model, gb_model, svm_model]
model_names = ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'Support Vector Machine']

for model, name in zip(models, model_names):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # Model evaluation metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=1)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"\n{name} Metrics:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1-Score: {f1}")
    print(f"ROC-AUC: {roc_auc}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve
    skplt.metrics.plot_roc(y_test, model.predict_proba(X_test))
    plt.title(f'{name} ROC Curve')
    plt.show()



# Step 6: Add a column for predicted Credit Worthiness to the original dataset using Random Forest model
data['Credit_Worthiness'] = rf_model.predict(features)

# Save the updated dataset to a new CSV file
data.to_csv('train_with_credit_worthiness.csv', index=False)
