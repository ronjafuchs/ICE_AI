import autokeras as ak
import pandas as pd

from keras.models import load_model
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelBinarizer

df = pd.read_csv("big_data.csv")
# print(df.columns)
desiredColumns = pd.read_csv("desired_columns.csv")
# print(desiredColumns.columns)
df.reset_index(drop=True, inplace=True)
df = df.reindex(columns=desiredColumns.columns)
# print(df.columns)
print(df["character_data_0_action"])
# print(df.columns)
print(df["character_data_0_action"].dtype)

# convert all non numeric columns to string to save headaches

df = df.astype(str)

print(df["character_data_0_action"])

allowed_values = [
    "STAND",
    "FORWARD_WALK",
    "DASH",
    "BACK_STEP",
    "CROUCH",
    "JUMP",
    "FOR_JUMP",
    "BACK_JUMP",
    "STAND_GUARD",
    "CROUCH_GUARD",
    "AIR_GUARD",
    "THROW_A",
    "THROW_B",
    "STAND_A",
    "STAND_B",
    "CROUCH_A",
    "CROUCH_B",
    "AIR_A",
    "AIR_B",
    "AIR_DA",
    "AIR_DB",
    "STAND_FA",
    "STAND_FB",
    "CROUCH_FA",
    "CROUCH_FB",
    "AIR_FA",
    "AIR_FB",
    "AIR_UA",
    "AIR_UB",
    "STAND_D_DF_FA",
    "STAND_D_DF_FB",
    "STAND_F_D_DFA",
    "STAND_F_D_DFB",
    "STAND_D_DB_BA",
    "STAND_D_DB_BB",
    "AIR_D_DF_FA",
    "AIR_D_DF_FB",
    "AIR_F_D_DFA",
    "AIR_F_D_DFB",
    "AIR_D_DB_BA",
    "AIR_D_DB_BB",
    "STAND_D_DF_FC",
]
print(len(allowed_values))
print(df["character_data_0_action"].unique())
df = df[df["character_data_0_action"].isin(allowed_values)]
print(df["character_data_0_action"].unique())

# Calculate the counts of each class
class_counts = df["character_data_0_action"].value_counts()

# Find classes with only one instance
classes_to_augment = class_counts[class_counts <= 3].index

# Duplicate rows of classes with only one instance
rows_to_add = []
for cls in classes_to_augment:
    for i in range(4):
        rows_to_add.append(df[df["character_data_0_action"] == cls])

# Concatenate the original df with the rows to add
if rows_to_add:  # Only concat if there's something to add
    df_augmented = pd.concat([df] + rows_to_add, ignore_index=True)
else:
    df_augmented = df.copy()

df = df_augmented

print(df["character_data_0_action"].value_counts())

y = df["character_data_0_action"]
X = df.drop(columns=["character_data_0_action"])
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)
print(X.shape)
print(y.shape)
# binarize the target variable

label_binarizer = LabelBinarizer()
label_binarizer.fit(allowed_values)
print(label_binarizer.classes_)
y = label_binarizer.transform(y)


print(X)
print(y)

# split into train and test sets
# X_train.info(verbose=True)
# y_train.info(verbose=True)
# bla = tf.convert_to_tensor(list(X_train))

# print(X_train)

# initialize the classifier

clf = ak.StructuredDataClassifier(
    max_trials=100,
    project_name="bigger_mctsai_model",
    metrics=["accuracy"],
    objective="val_accuracy",
    seed=42,
)


# split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)

clf.fit(X_train, y_train)


# evaluate the classifier
print(clf.evaluate(X_test, y_test))

# export as a keras model
model = clf.export_model()
model.summary()


# save the keras model
try:
    model.save("bigger_mctsai_model_autokeras")
except Exception:
    model.save("model_autokeras.h5")


loaded_model = load_model("bigger_mctsai_model_autokeras")
[print(i.shape, i.dtype) for i in loaded_model.inputs]
[print(o.shape, o.dtype) for o in loaded_model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in loaded_model.layers]
predict = loaded_model.predict(X_test)

df.head(1).to_csv("kek.csv", index=False, header=True)

# check if predict is equal to y_test
# np.testing.assert_array_equal(predict, y_test.astype(int))
print(predict)

test = label_binarizer.inverse_transform(predict)
print(test)

result = zip(
    test, pd.DataFrame(y_test, columns=label_binarizer.classes_).idxmax(axis=1)
)

# compute how many predictions were correct
correct = 0
for i in result:
    if i[0] == i[1]:
        correct += 1
print(correct / len(test))
