import autokeras as ak
import pandas as pd
from sklearn.model_selection import train_test_split


cls = ak.StructuredDataClassifier(project_name="category_test")

df = pd.read_csv("big_data.csv")
df = df.astype(str)

X = df.drop("character_data_0_action", axis=1)
y = df["character_data_0_action"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)

cls.fit(x=X_train, y=y_train)

print(cls.evaluate(X_test, y_test))