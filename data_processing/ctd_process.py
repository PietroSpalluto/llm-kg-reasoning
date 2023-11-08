import numpy as np
import pandas as pd

with open('../data/PrimeKG/ctd/CTD_exposure_events.csv', 'r') as f:
    lines = f.readlines()

field_next = False
for line in lines:
    if line.startswith('# Fields'):
        field_next = True
        continue
    if field_next:
        fields = line
        break

cols = fields[2:-2].split(',')

with open('../data/PrimeKG_processed/ctd/exposure_data.csv', 'w') as f:
    f.write(','.join(cols)+'\n')
    for line in lines:
        if not line.startswith('#'):
            f.write(line+'\n')


(pd.read_csv('../data/PrimeKG_processed/ctd/exposure_data.csv', index_col=False)
 .to_csv('../data/PrimeKG_processed/ctd/exposure_data.csv', index=False))
