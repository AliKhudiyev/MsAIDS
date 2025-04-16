import pandas as pd
import sys


input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_json(input_path)
print(df.shape)
db_csv = []
# df.rename(columns={kj
for i, column in enumerate(df.columns):
    vals = df[column].values
    for pair in vals:
        sponsor = pair[0]
        attendee = pair[1]
        db_csv.append([i, attendee, sponsor])
# print(db_csv)
df_new = pd.DataFrame(db_csv, columns=['slot', 'attendee', 'sponsor'])
print(df_new.head())
df_new.to_csv(output_path, index=None)

