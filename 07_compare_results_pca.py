import matplotlib.pyplot as plt
import pandas as pd
from config import OUTPUT_DIR,PLOTS_DIR

def main():
    paths=[OUTPUT_DIR/'clinical_results_pca.csv',OUTPUT_DIR/'lifestyle_results_pca.csv',OUTPUT_DIR/'merged_results_pca.csv']
    missing=[str(p) for p in paths if not p.exists()]
    if missing: raise FileNotFoundError('Missing result files:\n'+'\n'.join(missing))
    results=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    results.to_csv(OUTPUT_DIR/'all_results_pca.csv',index=False,encoding='utf-8-sig')
    f1=results.pivot(index='Model',columns='Dataset',values='F1').reset_index()
    f1['Improvement_vs_Clinical']=f1['Merged']-f1['Clinical']; f1['Improvement_vs_Lifestyle']=f1['Merged']-f1['Lifestyle']
    f1.to_csv(OUTPUT_DIR/'f1_comparison_pca.csv',index=False,encoding='utf-8-sig')
    for metric in ['Accuracy','Precision','Recall','F1','ROC_AUC']:
        pivot=results.pivot(index='Model',columns='Dataset',values=metric)
        ax=pivot.plot(kind='bar',figsize=(10,6),color=['#F4A261','#E76F51','#B22222'])
        ax.set_title(f'{metric}: Clinical vs Lifestyle vs Merged (PCA)'); ax.set_xlabel('Model'); ax.set_ylabel(metric); ax.set_ylim(0,1); ax.tick_params(axis='x',rotation=0); ax.grid(axis='y',linestyle='--',alpha=.3)
        plt.tight_layout(); plt.savefig(PLOTS_DIR/f'pca_comparison_{metric.lower()}.png',dpi=300); plt.close()
    results[['Dataset','Model','Original_Features','PCA_Components','Explained_Variance']].to_csv(OUTPUT_DIR/'pca_components_summary.csv',index=False,encoding='utf-8-sig')
    print(f1.round(4).to_string(index=False))

if __name__=='__main__': main()
