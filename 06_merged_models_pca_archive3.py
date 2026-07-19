import pandas as pd
from config_archive3 import OUTPUT_DIR
from data_utils_archive3 import load_original_data
from training_utils_archive3 import create_shared_split,evaluate_dataset

def main():
    full=load_original_data(); path=OUTPUT_DIR/'merged_dataset.csv'
    if not path.exists(): raise FileNotFoundError('Run 05_fusion_archive3.py first.')
    df=pd.read_csv(path); train_ids,test_ids=create_shared_split(full)
    results,roc=evaluate_dataset('Merged',df,train_ids,test_ids)
    pd.DataFrame(results).to_csv(OUTPUT_DIR/'merged_results_pca.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for model,vals in roc.items():
        for fpr,tpr in zip(vals['fpr'],vals['tpr']): rows.append({'Model':model,'FPR':fpr,'TPR':tpr,'AUC':vals['auc']})
    pd.DataFrame(rows).to_csv(OUTPUT_DIR/'merged_roc_curves_pca.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__': main()
