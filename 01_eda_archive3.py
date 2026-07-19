from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from config_archive3 import PLOTS_DIR, TARGET_COLUMN
from data_utils_archive3 import load_original_data

EDA_DIR = PLOTS_DIR / "eda"
EDA_DIR.mkdir(parents=True, exist_ok=True)

DARK_RED = "#B22222"
ORANGE = "#F28E2B"
LIGHT_ORANGE = "#F4A261"
PALE_ORANGE = "#FDE2C5"
DARK_BROWN = "#7F5539"

def safe_filename(name):
    return name.lower().replace(" ","_").replace("/","_").replace("\\","_")

def main():
    df = load_original_data()
    print(df.shape)
    print(df.isna().sum())
    print(df.describe(include='all').transpose())
    numeric = df.select_dtypes(include='number').columns.tolist()
    categorical = df.select_dtypes(exclude='number').columns.tolist()
    rows=[]
    for c in numeric:
        if c=='patient_id': continue
        v=df[c].dropna()
        rows.append({'Variable':c,'Count':len(v),'Mean':v.mean(),'Standard Deviation':v.std(),'Median':v.median(),'Minimum':v.min(),'Maximum':v.max(),'Missing':df[c].isna().sum()})
    pd.DataFrame(rows).to_csv(EDA_DIR/'numeric_summary_statistics.csv',index=False,encoding='utf-8-sig')
    for c in numeric:
        if c in ['patient_id',TARGET_COLUMN]: continue
        v=df[c].dropna()
        if v.empty or v.nunique()<=5: continue
        mean,med,std=v.mean(),v.median(),v.std()
        fig,ax=plt.subplots(figsize=(10,6))
        ax.hist(v,bins=30,color=LIGHT_ORANGE,edgecolor='white',alpha=.9)
        ax.axvline(mean,color=DARK_RED,linewidth=2.5,linestyle='--',label=f'Mean = {mean:.2f}')
        ax.axvline(med,color=ORANGE,linewidth=2.3,linestyle='-.',label=f'Median = {med:.2f}')
        ax.axvspan(mean-std,mean+std,color=PALE_ORANGE,alpha=.35,label=f'Mean ± SD ({std:.2f})')
        ax.set_title(f'Distribution of {c}')
        ax.set_xlabel(c); ax.set_ylabel('Number of observations'); ax.grid(axis='y',linestyle='--',alpha=.3); ax.legend(frameon=False)
        plt.tight_layout(); plt.savefig(EDA_DIR/f'distribution_{safe_filename(c)}.png',dpi=300); plt.close()
    for c in categorical:
        counts=df[c].fillna('Missing').value_counts()
        fig,ax=plt.subplots(figsize=(9,6))
        bars=ax.bar(counts.index.astype(str),counts.values,color=ORANGE,edgecolor=DARK_BROWN)
        ax.set_title(f'Distribution of {c}'); ax.set_xlabel(c); ax.set_ylabel('Number of patients'); ax.tick_params(axis='x',rotation=45); ax.grid(axis='y',linestyle='--',alpha=.3)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2,b.get_height(),f'{int(b.get_height())}',ha='center',va='bottom')
        plt.tight_layout(); plt.savefig(EDA_DIR/f'bar_{safe_filename(c)}.png',dpi=300); plt.close()
    counts=df[TARGET_COLUMN].value_counts().sort_index()
    fig,ax=plt.subplots(figsize=(8,6))
    bars=ax.bar(counts.index.astype(str),counts.values,color=[LIGHT_ORANGE,DARK_RED],edgecolor=DARK_BROWN)
    ax.set_title('Distribution of Heart Disease'); ax.set_xlabel('Heart Disease'); ax.set_ylabel('Number of patients'); ax.grid(axis='y',linestyle='--',alpha=.3)
    for b in bars:
        h=b.get_height(); ax.text(b.get_x()+b.get_width()/2,h,f'{int(h)}\n({h/len(df)*100:.1f}%)',ha='center',va='bottom',fontweight='bold')
    plt.tight_layout(); plt.savefig(EDA_DIR/'target_distribution.png',dpi=300); plt.close()
    corr_df=df.select_dtypes(include='number').drop(columns=['patient_id'],errors='ignore')
    corr=corr_df.corr(); corr.to_csv(EDA_DIR/'correlation_matrix.csv',encoding='utf-8-sig')
    fig,ax=plt.subplots(figsize=(13,11)); im=ax.imshow(corr,cmap='YlOrRd',vmin=-1,vmax=1,aspect='auto'); plt.colorbar(im,ax=ax,label='Correlation')
    ax.set_xticks(range(len(corr.columns))); ax.set_xticklabels(corr.columns,rotation=90); ax.set_yticks(range(len(corr.index))); ax.set_yticklabels(corr.index)
    for r in range(len(corr.index)):
        for c in range(len(corr.columns)):
            val=corr.iloc[r,c]; ax.text(c,r,f'{val:.2f}',ha='center',va='center',fontsize=8,color='white' if val>.6 else 'black')
    ax.set_title('Correlation Matrix'); plt.tight_layout(); plt.savefig(EDA_DIR/'correlation_matrix.png',dpi=300); plt.close()
    print(f'EDA saved in {EDA_DIR.resolve()}')

if __name__=='__main__': main()
