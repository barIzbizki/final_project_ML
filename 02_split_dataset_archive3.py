from config_archive3 import CLINICAL_COLUMNS, ID_COLUMN, LIFESTYLE_COLUMNS, OUTPUT_DIR, TARGET_COLUMN
from data_utils_archive3 import load_original_data

def main():
    df=load_original_data()
    clinical=df[[ID_COLUMN]+CLINICAL_COLUMNS+[TARGET_COLUMN]].copy()
    lifestyle=df[[ID_COLUMN]+LIFESTYLE_COLUMNS+[TARGET_COLUMN]].copy()
    clinical.to_csv(OUTPUT_DIR/'clinical_dataset.csv',index=False,encoding='utf-8-sig')
    lifestyle.to_csv(OUTPUT_DIR/'lifestyle_dataset.csv',index=False,encoding='utf-8-sig')
    print(clinical.shape,lifestyle.shape)

if __name__=='__main__': main()
