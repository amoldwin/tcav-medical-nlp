import json 
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import seaborn as sns

def sort_df(df,col, order):
    sorter= order
    if col =='index':
        df['index']=list(df.index)
    sorterIndex = dict(zip(sorter, range(len(sorter))))
    df['rank'] = df[col].map(sorterIndex)
    df[col].unique()
    df=df.sort_values(by='rank', kind='stable', ascending=True)[[col for col in df.columns if not col[0]=='rank' and not col[0]=='index']]
    return df


def return_cav_accuracies(cav_dir, codes):
    concepts = []
    for code in codes:
        concepts = concepts + [fn for fn in os.listdir(cav_dir) if code in fn and 'negative_'+code[len('positive_'):] in fn]
    print(codes)
    cav_accuracies = {}
    for concept in concepts:
        dict = pickle.load(open(os.path.join(cav_dir,concept),'rb'))
        positive_concept = concept.split('-allnegative')[0]
        if not positive_concept in cav_accuracies.keys():
            cav_accuracies[positive_concept] = []
        cav_accuracies[positive_concept].append(dict['accuracies']['overall'])

    concept_accs = {k:np.mean(v) for k,v in cav_accuracies.items() if not 'negative' in k}
    return concept_accs

def return_cav_metrics(cav_dir, codes):
    concepts = []
    for code in codes:
        concepts = concepts + [fn for fn in os.listdir(cav_dir) if code in fn and 'negative_'+code[len('positive_'):] in fn]

    cav_metrics = {}
    for concept in concepts:
        dict = pickle.load(open(cav_dir+concept,'rb'))
        positive_concept = concept.split('-negative')[0]
        if not positive_concept in cav_metrics.keys():
            cav_metrics[positive_concept] = []
        cav_metrics[positive_concept].append(dict['metrics'])
        
    return cav_metrics

def plot_multiple(targets,scores_dir,codes, names=None, cav_dir=None, normalized=True, metric='acc'):
    means = []
    errs=[]
    concepts = []
    medians = []
    i_ups=[]
    median_errs = []
    tcav_errs=[]
    dirs = []
    accs=[]
    for i,target in enumerate(targets):
        epoch_scores = []
        fn=target
        concepts.append(names[i])
        try:
            dicts  = pickle.load(open(os.path.join(scores_dir,target),'rb'))
        except:
            print('file not found',os.path.join(scores_dir,target))  
            [x.append(0) for x in [means, errs,medians, i_ups, median_errs, tcav_errs, dirs, accs]]
            continue
        
        means.append(-1*np.mean([dct['dirs'] for dct in dicts]))
        medians.append(-1*np.median([dct['dirs'] for dct in dicts]))
        if 'cav_accs' in dicts[0].keys():
            accs.append(np.mean([[v for k,v in dct['cav_accs'].items() if 'negative' not in k] for dct in dicts]))
        i_ups.append(np.mean([dct['i_up'] for dct in dicts]))
        errs.append(np.std([np.mean(dct['dirs']) for dct in dicts]))
        tcav_errs.append(np.std([np.mean(dct['i_up']) for dct in dicts]))

        median_errs.append(np.std([dct['val_directional_dirs_median'] for dct in dicts]))

        
  
    
    if normalized:
        means = [x/max(np.abs(means)) for x in means]
        errs = [x/max(np.abs(means)) for x in errs]

        medians = [x/max(np.abs(medians)) for x in medians]
        median_errs = [x/max(np.abs(medians)) for x in median_errs]
    
    if not cav_dir is None:
        if metric=='acc':
            accs = return_cav_accuracies(cav_dir+'/cavs/', codes)
            accs = [accs[k] if k in accs.keys() else 0  for k in codes ]
        elif metric=='f1':
            metrics_dcts = return_cav_metrics(cav_dir+'/cavs/', codes)
            accs = [np.mean([eval(x.split()[12])  for x in metrics_dcts[code] ])  for code in codes if code in metrics_dcts.keys()]
        
    return {'medians':medians, 'i_ups':i_ups, 'median_errs':median_errs, 'tcav_errs':tcav_errs, 'accs':accs}

def css_bar(start: float, end: float, color: str, row_idx:int, col:str,errs) -> str:
            """
            Generate CSS code to draw a bar from start to end.
            """
            err= errs[col[0]].iloc[row_idx]['Stdev']
            width=100
            css = "width: 10em; height: 80%;"
            err_start=max(0,end-err*100)
            err_end=min(100,end+err*100)
            if end > start:
                css += "background: "
                if start > 0:
                    css += " linear-gradient(90deg,"
                    css += f" transparent {err_start:.1f}%, red {err_start:.1f}%, "
                e = min(err_end, 100)
                css += f"red {e:.1f}%, transparent {e:.1f}%),"
                if start > 0:
                    css += f"linear-gradient(90deg, transparent {start:.1f}%, {color} {start:.1f}%, "
                e = min(end, width)
                css += f"{color} {e:.1f}%, transparent {e:.1f}%);"
                
               
                css+= 'background-size: 80% 10%, 80% 80% ; background-repeat: no-repeat, no-repeat; background-position: 0 20%, 0 0 ;'
                
            return css
def tcav_plot(model_names, targets,order, concept_names,metric='acc', ordered_labels=None):
#     def sort_df(df,col, order):
#             sorter= order
#             if col =='index':
#                 df['index']=list(df.index)
#             sorterIndex = dict(zip(sorter, range(len(sorter))))
#             df['rank'] = df[col].map(sorterIndex)
#             df[col].unique()
#             df=df.sort_values(by='rank', kind='stable', ascending=True)[[col for col in df.columns if not col[0]=='rank' and not col[0]=='index']]
#             return df
    df_pd = pd.DataFrame([zip(dct['i_ups'], dct['tcav_errs'], dct['accs']) for dct in targets], 
                              index=model_names,columns=concept_names).transpose()

    data=list(df_pd.apply(lambda row: np.reshape([list(tup) for tup in row], (3*len(model_names) )),axis=1))

    rows = concept_names
    cols=model_names
    sub_cols = ['TCAV','Stdev',"CAV\nAccuracy"]
    mux = pd.MultiIndex.from_product([cols, sub_cols])
    df_pd = pd.DataFrame(data, columns=mux, index=rows).round(2)
    df_pd=df_pd[model_names]

    pd.set_option('precision',2)
    df_pd['long_title']=df_pd.apply(lambda row: row.name,axis=1 )
    df_pd['concept']=df_pd['long_title'].apply(lambda x:  x)
    df_pd=df_pd.groupby('concept').apply(np.mean)
    df_pd=df_pd.reset_index()
    df_pd = sort_df(df_pd, 'concept',order)

    df_pd.index=df_pd['concept']
    df_pd=df_pd[[col for col in df_pd.columns if col[0] not in ['rank','concept']]]
    df_pd=df_pd[model_names]
    # df.style.background_gradient(cmap ='Reds')\
    #         .set_properties(**{'font-size': '12px'})
    no_error_df_pd = df_pd[[col for col in df_pd.columns if col[1]!='Stdev']]
    no_error_df_pd.index.name=None
    
    no_error_df_pd = sort_df(no_error_df_pd, 'index',order)
    if ordered_labels:
        no_error_df_pd.index=ordered_labels
        no_error_df_pd.index.name = 'Concept'
        
        
    no_error_df_pd[('','Concept')] = no_error_df_pd.index
    no_error_df_pd = no_error_df_pd[[('','Concept')]+[x for x in no_error_df_pd.columns if not 'Concept' in x ]]
    no_error_df_pd = no_error_df_pd.reset_index(drop=True)
    
    def add_errorbars(x):
        return np.where(x == np.nanmax(x.to_numpy()), "{height: 1px;background: black;}", None)
    styled=no_error_df_pd.style.format({'vm_unity': '{:.2%}'})\
    .bar(subset = [col for col in df_pd.columns if col[1]=='TCAV'], align='mid', color=['#d65f5f', '#5fba7d'], vmax=1)\
    .bar(subset = [col for col in df_pd.columns if col[1]=="CAV\nAccuracy"], align='mid', color=['#26abff', '#26abff'],vmax=1)\
    .set_properties(**{'max-width': '5000px', 'font-size': '10pt'})\
    .set_properties(subset=[('','Concept')], **{'width': '3000px'})\
    .apply(lambda x:  [css_bar(start=0.1,end=y*100, color='#5fba7d', row_idx=i,
                               col=x.name, 
                               errs=df_pd[[col for col in df_pd.columns if col[1]=='Stdev']]) for i,y in enumerate(x)], subset=[col for col in df_pd.columns if col[1]=='TCAV'])
    pd.options.display.max_rows=100
#     pd.set_option("display.max_colwidth",100)
#     pd.set_option('display.width', 10000)

#     pd.set_option("display.latex.multicolumn_format","True")
#     pd.set_option("display.expand_frame_repr",False)
    return styled