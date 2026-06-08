import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('../dataset/mumbai_housing.csv')

df = pd.get_dummies(df)

X = df.drop("Price", axis=1)
y = df["Price"]

# save feature columns
pickle.dump(X.columns, open("columns.pkl","wb"))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(n_estimators=200)

model.fit(X_train, y_train)

pickle.dump(model, open("model.pkl","wb"))

print("Model trained successfully")