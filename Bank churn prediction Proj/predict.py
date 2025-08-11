
import joblib
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder

# load pipeline and dataset
pipeline = joblib.load("notebooks/bank_churn_pipeline.pkl")
df = pd.read_csv("data/Churn_Modelling.csv")

# Picking random row
rnd = random.choice(df.index)
ex = df.loc[[rnd]]

print("\nCustomer:")
print(ex)

# Converting gender and Dropping target
le = LabelEncoder()
ex['Gender'] = le.fit_transform(ex['Gender'])
test_row = ex.drop(columns=['Exited'])

# prediction
pred = pipeline.predict(test_row)[0]
prob = pipeline.predict_proba(test_row)[0][1]

# Result
print("\n Prediction:", "Will leave" if pred == 1 else "Will stay")
print(f" Probability: {prob:.2%}")
