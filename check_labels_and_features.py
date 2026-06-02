import numpy as np
import pandas as pd
import os
base='mimic_processed'
print('CWD:', os.getcwd())
# Labels
for fn in ['y_train.npy','y_test.npy','y_train_raw.npy','y_train_smote.npy']:
    p=os.path.join(base,fn)
    if os.path.exists(p):
        arr=np.load(p)
        print(fn, arr.shape, 'mean=', float(arr.mean()), 'counts=', (arr.astype(int).sum(), len(arr)-arr.astype(int).sum()))
    else:
        print(fn, 'MISSING')
# Features
fpath=os.path.join(base,'feature_names_final.csv')
if os.path.exists(fpath):
    df=pd.read_csv(fpath)
    feats=df['feature'].astype(str).tolist()
    keywords=['vaso','intub','dialysis','death','vent','peep','fio2','vaso_max_rate','vaso_flag']
    found=[f for f in feats if any(k in f.lower() for k in keywords)]
    print('\nFeatures matching leak-suspect keywords:')
    print('\n'.join(found[:50]))
else:
    print('feature_names_final.csv MISSING')
