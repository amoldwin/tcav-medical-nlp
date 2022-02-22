# coding=utf-8
# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""TODO: Add a description here."""

from __future__ import absolute_import, division, print_function

import csv
import os
import pandas as pd

import datasets
from dateutil import parser as date_parser

# TODO: Add BibTeX citation
# Find for instance the citation on arxiv or on the dataset repo/website

_CITATION = """\
@InProceedings{huggingface:dataset,
title = {Mortality prediction based on discharge notes from MIMIC-III},
authors={huggingface, Inc.
},
year={2020}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This is the mortality task explored in our paper on concept-level interpretability of models that process clinical notes. 
"""

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = "https://github.com/amoldwin/Health-BERT-TCAV"

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace dataset library don't host the datasets but only point to the original files
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
_URLs = {
    'first_domain': "https://github.com/amoldwin/Health-BERT-TCAV",
#     'second_domain': "https://huggingface.co/great-new-dataset-second_domain.zip",
}

CALIBRATION = datasets.NamedSplit("calibration")

# TODO: Name of the dataset usually match the script name with CamelCase instead of snake_case
class ClueDataset(datasets.GeneratorBasedBuilder):
    """TODO: Short description of my dataset."""

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')
    BUILDER_CONFIGS = [
       
        datasets.BuilderConfig(name="nicu_los", version=VERSION,
                               description="NICU LOS prediction based on admission notes from MIMIC-III without removing patients 2 days before the cutoff"),        
        datasets.BuilderConfig(name="CCS", version=VERSION,
                               description="CAD prediction based on admission notes from MIMIC-III"),
        datasets.BuilderConfig(name="CCS_txt_str", version=VERSION,
                               description="CAD prediction based on admission notes and first day structured data from MIMIC-III"),
        datasets.BuilderConfig(name="NER", version=VERSION,
                               description="NER prediction based on N2C2 2014 Data")
    ]
    
    
    
    
    DEFAULT_CONFIG_NAME = None

    def _info(self):
        self.filepaths={}
        features = { }
        if self.config.name == 'nicu_los':
            features['text'] = datasets.Value("string")
            features['nicu_los'] = datasets.Value("int8")
            self.filepaths['nicu_los']={'train': 'los_nicu_train_admissions.csv',
                                             'val': 'los_nicu_val_admissions.csv',
                                             'test': 'los_nicu_test_admissions.csv'}
        if self.config.name == 'CCS':
            features['text'] = datasets.Value("string")
            features['CCS'] = datasets.Value("int8")
            self.filepaths['CCS']={'train': 'los_metabolic_train_admissions.csv',
                                             'val': 'los_metabolic_val_admissions.csv',
                                             'test': 'los_metabolic_test_admissions.csv'}
        if self.config.name == 'CCS_txt_str':
            features['text'] = datasets.Value("string")
            features['CCS__txt_str'] = datasets.Value("int8")
            features['structured_features'] = datasets.Value("string")
            self.filepaths['CCS__txt_str']={'train': 'metabolic_train_plus_structured.csv',
                                             'val': 'metabolic_val_plus_structured.csv',
                                             'test': 'metabolic_test_plus_structured.csv'}
        if self.config.name == 'NER':
            features['input_ids'] = datasets.Value("string")
            features['NER'] = datasets.Value("string")
            self.filepaths['NER']={'train': 'NER_train.csv',
                                         'val': 'NER_val.csv',
                                        'test':'NER_val.csv' }
        features = datasets.Features(features)
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # TODO: This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLs
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive 
        # my_urls = _URLs[self.config.name]
        # data_dir = dl_manager.download_and_extract(my_urls)
        data_dir = self.config.data_dir

        # noinspection PyTypeChecker
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, self.filepaths[self.config.name]['train']),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, self.filepaths[self.config.name]['val']),
                },
            ),
            datasets.SplitGenerator(
                name=CALIBRATION,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, self.filepaths[self.config.name]['val']),
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": os.path.join(data_dir, self.filepaths[self.config.name]['test']),
                },
            ),

        ]

    def _get_label_parse_fn(self):

        elif self.config.name == ('nicu_los'):
            def parse_fn(data):
                return {f'nicu_los': int(float(data['label']))}
        elif self.config.name == 'CCS':
            def parse_fn(data):
                return {f'CCS': int(float(data['label']))}
       
        elif self.config.name == 'CCS_txt_str':
            def parse_fn(data):
                return {f'CCS_txt_str': int(float(data['label']))}
       
        elif self.config.name.startswith('NER'):
            def parse_fn(data):
                return {f'NER':data['label_ids']}
            def parse_fn(data):
                return {f'{disease}_stage': int(float(data['label']))}
        else:
            return None
        return parse_fn

    def _generate_examples(self, filepath):
        """ Yields examples. """
        # TODO: This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)
#         allowed_features = ['TEXT','label']

        data_df = pd.read_csv(filepath)
        data_df = data_df#[[col for col in data_df.columns if col in allowed_features]]
        label_parse_fn = self._get_label_parse_fn()
        for id_ in range(len(data_df)):
            row = data_df.iloc[id_]
            if 'chunked_tokens' in row.keys():
                #NER task
                ex = {'input_ids':row['chunked_tokens'],**label_parse_fn(row)}
            elif 'max_heightcm' in row.keys():
                #Structured data+text metabolic syndrome prediction
                str_lst = str(list(row[[x for x in row.index if 'MED' in x or 'dx' in x or 'min' in x or 'max' in x
                                        or 'mean' in x or 'stdev' in x or 'median' in x or 'count' in x or 'GENDER' in x or 'AGE' in x]]))
                ex = {'text':row['TEXT'],'structured_features':str_lst,**label_parse_fn(row)}
            else:   
                #Tasks using only text
                ex = {'text':row['TEXT'],**label_parse_fn(row)}
            yield id_, ex
        

print('Imported Data Loader')