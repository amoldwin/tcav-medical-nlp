import pandas as pd
import os
from tqdm import tqdm
from multiprocessing import Pool

dirs_dct = dict(list(pd.read_csv('../directory_paths.csv')['paths'].apply(eval)))

stays_df = pd.read_csv(os.path.join(dirs_dct['mimic_dir'],'ICUSTAYS.csv'))

adm_df = pd.read_csv(os.path.join(dirs_dct['mimic_dir'],'ADMISSIONS.csv'))

d_items_df = pd.read_csv(os.path.join(dirs_dct['mimic_dir'],'D_ITEMS.csv'))

d_labitems_df = pd.read_csv(os.path.join(dirs_dct['mimic_dir'],'D_LABITEMS.csv'))

len(set(d_labitems_df['ITEMID']).intersection(d_items_df['ITEMID']))

merge_mappings = pd.read_csv('merged_HADMs.csv')

adm_df = adm_df.merge(merge_mappings, on='HADM_ID', how='left')

# dict of form { field: list((relevant itemids),(allowed range)) }
# dict of form { field: list((relevant itemids),(allowed range)) }
fields_dct = {
    'gest_age':[(3446,), (0,30000)],
    'birth_weight':[(4183,3723),(0,40000)],
}
def extract_vals(chart_df):
    chart_df = chart_df.merge(merge_mappings, on='HADM_ID', how='left')

    for field, lst in fields_dct.items():
#         print(field)        
        chart_df.loc[chart_df.apply(lambda row: row['ITEMID'] in lst[0] , axis=1),'field']=field

    chart_df=chart_df.dropna(subset=['field'])
    return chart_df

chunksize = 100000
chart_path = os.path.join(dirs_dct['mimic_dir'],'CHARTEVENTS.csv')
lab_path = os.path.join(dirs_dct['mimic_dir'],'LABEVENTS.csv')
prescriptions_path = os.path.join(dirs_dct['mimic_dir'],'PRESCRIPTIONS.csv')
res_df = pd.DataFrame()
with Pool(30) as p:
    print('starting charts', flush=True)
    chart_res_df = pd.concat(p.map(extract_vals,pd.read_csv(chart_path, chunksize=chunksize)), ignore_index=True)
    chart_res_df.to_csv(os.path.join(dirs_dct['data_dir'],'all_nicu_chart_stats_merged.csv'))
    del chart_res_df
    