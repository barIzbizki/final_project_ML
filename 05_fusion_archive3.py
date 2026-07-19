import pandas as pd
from config_archive3 import ID_COLUMN,OUTPUT_DIR,TARGET_COLUMN

def main():
    c=OUTPUT_DIR/'clinical_dataset.csv'; l=OUTPUT_DIR/'lifestyle_dataset.csv'
    if not c.exists() or not l.exists(): raise FileNotFoundError('Run 02_split_dataset_archive3.py first.')
    clinical=pd.read_csv(c); lifestyle=pd.read_csv(l)
    merged=clinical.merge(lifestyle.drop(columns=[TARGET_COLUMN]),on=ID_COLUMN,how='inner',validate='one_to_one')
    merged.to_csv(OUTPUT_DIR/'merged_dataset.csv',index=False,encoding='utf-8-sig')
    print(merged.shape)

if __name__=='__main__': main()
