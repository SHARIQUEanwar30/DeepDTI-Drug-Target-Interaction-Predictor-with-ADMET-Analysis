from tdc.multi_pred import DTI
import pandas as pd

def load_data(sample_size=2000):

    data = DTI(name='BindingDB_Kd')
    data.harmonize_affinities(mode='max_affinity')

    df = data.get_data()

    threshold = 1000
    df['interaction'] = df['Y'].apply(lambda x: 1 if x < threshold else 0)

    df = df[['Drug','Target','interaction']]
    df = df.sample(sample_size, random_state=42)

    return df
