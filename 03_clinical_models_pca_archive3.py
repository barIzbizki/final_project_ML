import pandas as pd
from config_archive3 import OUTPUT_DIR
from data_utils_archive3 import load_original_data
from training_utils_archive3 import create_shared_split,evaluate_dataset

def main():
    full=load_original_data(); path=OUTPUT_DIR/'clinical_dataset.csv'
    if not path.exists(): raise FileNotFoundError('Run 02_split_dataset_archive3.py first.')
    df=pd.read_csv(path); train_ids,test_ids=create_shared_split(full)
    results,_=evaluate_dataset('Clinical',df,train_ids,test_ids)
    pd.DataFrame(results).to_csv(OUTPUT_DIR/'clinical_results_pca.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__': main()
