from flaml import AutoML
import pandas as pd

df = pd.read_csv("../testdata/big_data.csv")
X_train = df.drop("character_data_0_action", axis=1)
y_train = df["character_data_0_action"]

automl = AutoML()
automl.fit(X_train, y_train, task="classification")

print(automl.best_config)
print(automl.best_estimator)