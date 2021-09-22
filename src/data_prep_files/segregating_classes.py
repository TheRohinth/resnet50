import json
import pandas as pd

paths = json.loads(open('paths.json',).read())
df = pd.read_csv(paths['train_csv'])
print(df.loc[df['opacity'] == 1])