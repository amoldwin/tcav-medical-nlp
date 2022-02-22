import pandas as pd
import os
import numpy as np
# from brat_parser import get_entities_relations_attributes_groups
import gzip
import time
t=time.time()
print(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
def get_brat_sections():
    sections_dict = {}
    file_list = []
    sectionspath = '../../LHC_mimic/mimic3_1.4/notes/sections/'
    for hashdir in os.listdir(sectionspath):
        for subjectdir in os.listdir(sectionspath+hashdir):
            for admissiondir in os.listdir(sectionspath+hashdir+'/'+subjectdir):
                for fn in os.listdir(sectionspath+hashdir+'/'+subjectdir+'/'+admissiondir):
                    ann_fn = sectionspath+hashdir+'/'+subjectdir+'/'+admissiondir + '/' + fn
#                     entities, relations, attributes, groups = get_entities_relations_attributes_groups(ann_fn)
                    file_list.append(ann_fn)
    return file_list
file_list = get_brat_sections()

from multiprocessing import Pool, Process, Manager
def read_ann(file):
    start = time.time()
    with gzip.open(file, 'rb') as ann_file:
        text = ann_file.read()
    return (file,text)
start = time.time()
pool = Pool(50)
collection = pool.map(read_ann, file_list)
pool.close()
pool.join()   

pd.DataFrame(collection).to_csv('all_section_annotations.csv')