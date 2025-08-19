import pandas as pd
import joblib
import random

# Load pipeline 
model = joblib.load("notebooks/fraud_detection_pipeline.pkl")

# Load dataset
df = pd.read_csv("data/fraudTest.csv")

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['dayofweek'] = df['trans_date_trans_time'].dt.dayofweek
df['dob'] = pd.to_datetime(df['dob'])
df['age'] = df['trans_date_trans_time'].dt.year - df['dob'].dt.year

drop_cols = ['trans_date_trans_time','dob','cc_num','first','last',
             'street','city','state','zip','lat','long','job','trans_num']
df = df.drop(columns=drop_cols)

# Pick a row
rnd_idx = random.randint(0, len(df)-1)
row = df.iloc[[rnd_idx]]

# Separate features and target
X_row = row.drop(columns=['is_fraud'])
y_true = row['is_fraud'].values[0]

#  Predict
y_pred = model.predict(X_row)[0]
y_proba = model.predict_proba(X_row)[0][1]

# results
print("Randomly selected transaction:")
print(row)

print("\nIs fraud:", y_true)
print("Model prediction:", y_pred)
print("Fraud probability:", round(y_proba, 4))
