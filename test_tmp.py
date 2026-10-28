import pandas as pd
import json
from backend.app import data_processing
df = pd.read_csv('data/uploads/2c5279a4396e1f58.csv')
print('shape:', df.shape)
qr = data_processing.quality_report(df)
print(json.dumps(qr, indent=2))
