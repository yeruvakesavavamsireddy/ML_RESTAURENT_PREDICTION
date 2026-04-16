import numpy as np
import pandas as pd

def engineer_features(df):
    if 'Votes' in df.columns:
        df['Votes'] = pd.to_numeric(df['Votes'], errors='coerce').fillna(0)
        df['Votes_log'] = np.log1p(df['Votes'])

    if 'Average Cost for two' in df.columns:
        df['Cost_log'] = np.log1p(df['Average Cost for two'])

    return df