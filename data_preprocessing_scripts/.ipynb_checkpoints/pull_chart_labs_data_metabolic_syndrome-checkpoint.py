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

merge_mappings = pd.read_csv(os.path.join(dirs_dct['data_dir'],'merged_HADMs.csv'))

adm_df = adm_df.merge(merge_mappings, on='HADM_ID', how='left')

# dict of form { field: list((relevant itemids),(allowed range)) }
# dict of form { field: list((relevant itemids),(allowed range)) }
fields_dct = {
    'heartrate':[(211,220045), (0,300)],
    'sysbp':[(51,442,455,6701,220179,220050),(0,400)],
    'diasbp':[(8368,8440,8441,8555,220180,220051),(0,300)],
    'meanbp':[(456,52,6702,443,220052,220181,225312),(0,300)],
    'resprate':[(615,618,220210,224690),(0,70)],
    'tempf': [(223761,678), (70,120)],
    'tempc': [(223762,676) , (10,50)],
    'spo2': [(646,220277),(0,101)],
    'glucose': [(807,811,1529,3745,3744,225664,220621,226537,50809,50931), (0,5000)],
    'weightlb' : [(3581,),(0,974)],
    'weightoz' : [(3582,), (0,15591.1)],
    'weightkg' : [(762, 763, 3723, 3580),(0,442)],
    'heightin' : [(1394,920,4187,3486),(0,107)],
    'heightcm'  : [(3485,4188,),(0,272)],
    'cholesterol' :[(220603,),(0,3165)],
    'hdl' : [(220624,50904),(0,1000)],
    'ldl': [(225671,227441,50905,50906),(0,3165)],
    'hema1c': [(50852,), (0,25)],
    'triglyceride':[(225693,),(0,1000)],
    'abd_girth': [(3294,),(0,1000)],
    'triglycerides': [(1540,225693), (0,1000)]
}
def extract_vals(chart_df):
    chart_df = chart_df.merge(merge_mappings, on='HADM_ID', how='left')

    for field, lst in fields_dct.items():
#         print(field)        
        chart_df.loc[chart_df.apply(lambda row: row['ITEMID'] in lst[0] and row['VALUENUM']>lst[1][0] and row['VALUENUM']<lst[1][1], axis=1),'field']=field


    chart_df.loc[chart_df['field']=='tempf','VALUENUM']=chart_df[chart_df['field']=='tempf']['VALUENUM'].apply(lambda x: (x-32)/1.8)
    chart_df.loc[chart_df['field']=='tempf','field']='tempc'

    chart_df.loc[chart_df['field']=='weightlb','VALUENUM']=chart_df[chart_df['field']=='weightlb']['VALUENUM'].apply(lambda x: x*0.453592)
    chart_df.loc[chart_df['field']=='weightlb','field']='weightkg'
    chart_df.loc[chart_df['field']=='weightoz','VALUENUM']=chart_df[chart_df['field']=='weightoz']['VALUENUM'].apply(lambda x: x*0.0283495)
    chart_df.loc[chart_df['field']=='weightoz','field']='weightkg'

    chart_df.loc[chart_df['field']=='heightin','VALUENUM']=chart_df[chart_df['field']=='heightin']['VALUENUM'].apply(lambda x: x*2.54)
    chart_df.loc[chart_df['field']=='heightin','field']='heightcm'

    chart_df = chart_df.dropna(subset=['VALUENUM','field','CHARTTIME'],how='any')
    if 'ERROR' in chart_df.columns:
        chart_df = chart_df[chart_df['ERROR']!=1]
    #Changed ICUSTAY_ID to HADM_ID
    chart_df = chart_df.merge(adm_df[['ADMITTIME','DISCHTIME','new_HADM']], on='new_HADM', how='left')

    chart_df['CHARTTIME']=pd.to_datetime(chart_df['CHARTTIME'])
    chart_df['ADMITTIME']=pd.to_datetime(chart_df['ADMITTIME'])
    if len(chart_df)==0:
        return pd.DataFrame()
    chart_df['time_since_in']=chart_df.apply(lambda row: (row['CHARTTIME']-row['ADMITTIME']), axis=1)

#     chart_df=chart_df[chart_df.apply(lambda row: (row['time_since_in'].days<1 and row['time_since_in'].days>=0) or row['field'] in ['weightkg', 'heightcm']    ,axis=1 )]

#     chart_df = chart_df.dropna(subset=['VALUENUM','field','time_since_in'], how='any')
    if len(chart_df)==0:
        return pd.DataFrame()
    chart_df = chart_df.groupby(['new_HADM','field'])['VALUENUM'].apply(lambda x: list(x)).unstack().reset_index()
    return chart_df


ace_lst = ['Benazepril', 'Lotensin', 'Captopril', 'Capoten', 'Enalapril','Enalaprilat', 'Vasotec', 'Fosinopril','Monopril', 'Lisinopril', 'Zestril','Prinivil', 'Moexipril','Univasc', 'Perindopril','Aceon', 'Quinapril', 'Accupril', 'Ramipril', 'Altace',  'Trandolapril', 'Mavik']
beta_lst = ['acebutolol','atenolol','Tenormin','betaxolol','bisoprolol','carvedilol','carvedilol','Coreg CR','labetalol','Trandate','metoprolol','Kapspargo', 'Toprol-XL','metoprolol tartrate','Lopressor',
    'nadolol','Corgard','nebivolol','Bystolic','pindolol','propranolol','Inderal', 'Inderal', 'InnoPran','timolol']
ins_lst = ['Admelog','lispro','Afrezza','insulin','Apidra','Solostar','glulisine','Fiasp','Flextouch', 'aspart','humalog','Novolog','Humulin','Novolin','Humulin','NPH','isophane','Novolin','Basaglar','glargine','Lantus','Lantus Solostar','glargine','Levemir','detemir','Toujeo',
    'glargine','Tresiba', 'FlexTouch','degludec','Humalog','novoLog','aspart','Ryzodeg','degludec']
fibr_lst = ['Fenofibrate', 'Tricor', 'Gemfibrozil','Lopid','Antara', 'Fenoglide','Lipofen','Triglide', 'Trilipix', 'Fenofibric Acid']
stat_lst = ['Lipitor','Lescol','Mevacor','Pravachol','Crestor','Zocor','atorvastatin','fluvastatin','lovastatin','Altoprev','Livalo','Pravachol',
           'Advicor','Simcor','Vytorin']


ace_lst = [x.lower() for x in ace_lst]
beta_lst = [x.lower() for x in beta_lst]
ins_lst = [x.lower() for x in ins_lst]
fibr_lst = [x.lower() for x in fibr_lst]
stat_lst = [x.lower() for x in stat_lst]


ace_fn = lambda x: str(x).lower().endswith('pril') or any([y in str(x).lower() for y in ace_lst])
beta_fn = lambda x: str(x).lower().endswith('lol')  or any([y in str(x).lower() for y in beta_lst])
stat_fn = lambda x: 'statin' in x or any([y in str(x).lower() for y in beta_lst])

ins_fn = lambda x: any([y in str(x).lower() for y in ins_lst])
fibr_fn = lambda x: any([y in str(x).lower() for y in fibr_lst])

drug_types = {'MED_ACE':ace_fn, 'MED_BETA':beta_fn, 'MED_INS':ins_fn, 'MED_FIBR':fibr_fn, 'MED_STAT':stat_fn}

def prescription_overlaps_first_day(row):
    return pd.Interval(row['STARTDATE'],row['ENDDATE']).overlaps(pd.Interval(row['ADMITTIME'], row['ADMITTIME']+pd.DateOffset(days=1)))

def extract_prescription_vals(scrips_df):
    scrips_df = scrips_df.merge(merge_mappings, on='HADM_ID', how='left')

    scrips_df = scrips_df.merge(adm_df[['ADMITTIME','DISCHTIME','new_HADM']], on='new_HADM', how='left')

    scrips_df['STARTDATE']=pd.to_datetime(scrips_df['STARTDATE'])
    scrips_df['ENDDATE']=pd.to_datetime(scrips_df['ENDDATE'])
    scrips_df['ADMITTIME']=pd.to_datetime(scrips_df['ADMITTIME'])

    scrips_df=scrips_df.dropna(subset=['ADMITTIME','STARTDATE','ENDDATE'],how='any')

    scrips_df['ADMITTIME'].iloc[0]+pd.DateOffset(days=1)


#     scrips_df = scrips_df[scrips_df.apply(lambda row: row['STARTDATE']<row['ENDDATE']  ,axis=1)]
#     scrips_df = scrips_df[scrips_df.apply(lambda row: prescription_overlaps_first_day(row)  ,axis=1)]
    if len(scrips_df)==0:
        return pd.DataFrame()
    for field,fn in drug_types.items():
        scrips_df.loc[scrips_df.apply(lambda row: fn(str(row['DRUG'])) or fn(str(row['DRUG_NAME_POE']))or fn(str(row['DRUG_NAME_GENERIC'])) , axis=1),'field']=field


    scrips_df= scrips_df.dropna(subset=['field'])

    if len(scrips_df)==0:
        return pd.DataFrame()
    scrips_result_df = scrips_df.groupby(['new_HADM','field'])['field'].apply(lambda x: 1 if len(x)!=0 else 0).unstack().reset_index()
    
    return scrips_result_df
#     result_df=result_df.merge(scrips_result_df,on='HADM_ID', how='outer')

def extract_diagnoses_vals():
    diagnoses_df = pd.read_csv(os.path.join(dirs_dct['mimic_dir'],'DIAGNOSES_ICD.csv'))
    diagnoses_df = diagnoses_df.merge(merge_mappings, on='HADM_ID', how='left')

    diabetes_codes = ['24900', '25000', '25001', '7902', '79021', '79022', '79029', '7915', '7916', 'V4585', 'V5391', 'V6546', '24901', '24910', '24911', '25002', '25003', '25010', '25011', '25012', '250', '24940', '24941', '25040', '25041', '25042', '25043', '24950', '24951', '25050', '25051', '25052', '25053', '24960', '24961', '25060', '25061', '25062', '25063', '24970', '24971', '25070', '25071', '25072', '25073', '24990', '24991', '25090', '25091', '24920', '24921', '24930', '24931', '24980', '24981', '25020', '25021', '25022', '25023', '25030', '25031', '25032', '25033', '25080', '25081', '25082', '25083', '25092', '25093']

    diagnoses_df= diagnoses_df[diagnoses_df['ICD9_CODE'].apply(lambda x: x in diabetes_codes)] 

    diagnoses_df['diabetesdx'] =1
    return diagnoses_df[['new_HADM','diabetesdx']]

chunksize = 100000
chart_path = os.path.join(dirs_dct['mimic_dir'],'CHARTEVENTS.csv')
lab_path = os.path.join(dirs_dct['mimic_dir'],'LABEVENTS.csv')
prescriptions_path = os.path.join(dirs_dct['mimic_dir'],'PRESCRIPTIONS.csv')
res_df = pd.DataFrame()
with Pool(30) as p:
    print('starting charts', flush=True)
    chart_res_df = pd.concat(p.map(extract_vals,pd.read_csv(chart_path, chunksize=chunksize)), ignore_index=True)
    chart_res_df.to_csv(os.path.join(dirs_dct['data_dir'],'all_admission_fulladm_chart_stats_merged.csv'))
    del chart_res_df
    print('starting labs', flush=True)
    lab_res_df = pd.concat(p.map(extract_vals,pd.read_csv(lab_path, chunksize=chunksize)), ignore_index=True)
    lab_res_df.to_csv(os.path.join(dirs_dct['data_dir'],'all_admission_fulladm_lab_stats_merged.csv'))
    del lab_res_df
    print('starting rx', flush=True)
    rx_res_df = pd.concat(p.map(extract_prescription_vals,pd.read_csv(prescriptions_path, chunksize=chunksize)), ignore_index=True)
    rx_res_df.to_csv(os.path.join(dirs_dct['data_dir'],'all_admission_fulladm_rx_stats_merged.csv'))
print('starting dx', flush=True)
dx_df = extract_diagnoses_vals()
dx_df.to_csv(os.path.join(dirs_dct['data_dir'],'all_admission_fulladm_dx_stats_merged.csv'))