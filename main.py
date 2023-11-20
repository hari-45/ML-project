import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Step 1: Load and preprocess the data
data = pd.read_csv('train.csv')
data.dropna(inplace=True)
label_encoder = LabelEncoder()
data['Payment_Behaviour'] = label_encoder.fit_transform(data['Payment_Behaviour'])
features = data[['Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Payment_Behaviour']]
target = data['Credit_Score']

# Step 2: Split the data and train the Random Forest model
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 3: Evaluate the model
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]


# Save the updated dataset to a new CSV file
data.to_csv('train_with_credit_worthiness.csv', index=False)
