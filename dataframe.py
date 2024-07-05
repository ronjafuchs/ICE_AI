import pandas as pd
import json

# Assuming the DataFrame is stored in a variable called df

# Preprocess the columns containing JSON strings
df = pd.read_csv('test.csv')
print(df)
json_columns = ['frameData.characterData', 'frameData.front']
for column in json_columns:
    df[column] = df[column].apply(json.loads)

# Flatten the DataFrame
df_flat = pd.json_normalize(df.to_dict('records'))

# Display the flattened DataFrame
print(df_flat)
