import pandas as pd
from config import OUTPUT_DIR
from data_utils import load_original_data
from training_utils import create_shared_split,evaluate_dataset

def main():
    full=load_original_data(); path=OUTPUT_DIR/'lifestyle_dataset.csv'
    if not path.exists(): raise FileNotFoundError('Run 02_split_dataset.py first.')
    df=pd.read_csv(path); train_ids,test_ids=create_shared_split(full)
    results,_=evaluate_dataset('Lifestyle',df,train_ids,test_ids)
    pd.DataFrame(results).to_csv(OUTPUT_DIR/'lifestyle_results_pca.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__': main()
