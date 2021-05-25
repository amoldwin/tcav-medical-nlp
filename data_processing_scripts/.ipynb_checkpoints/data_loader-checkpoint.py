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
        datasets.BuilderConfig(name="discharge_mortality", version=VERSION,
                               description="Mortality prediction based on discharge notes from MIMIC-III"),
    ]
    
    
    
    
    DEFAULT_CONFIG_NAME = None

    def _info(self):
        self.filepaths={}
        features = {
#             "hadm_id": datasets.Value("int32"),
#             "timestamp": datasets.Value("timestamp[s]"),
            "text": datasets.Value("string"),
#             "TEXT": datasets.Value("string"),
#             "label": datasets.Value("int8"),
        }
        
        multitask = self.config.name == 'multitask'
        if self.config.name == 'discharge_mortality':
            features['discharge_mortality'] = datasets.Value("int8")
            self.filepaths['discharge_mortality']={'train': 'mortality_train.csv',
                                             'val': 'mortality_val.csv',
                                             'test': 'mortality_test.csv'}
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
        if self.config.name.startswith('discharge_mortality'):
            def parse_fn(data):
                return {f'discharge_mortality': int(float(data['label']))}
            
            
        elif self.config.name.startswith('staging'):
            disease = self.config.name[8:]

            def parse_fn(data):
                return {f'{disease}_stage': int(float(data['label']))}

        elif self.config.name == 'phenotyping':
            def parse_fn(data):
                phenotypes = [bool(data[f'{phenotype}']) for phenotype in constants.PHENOTYPE_LABELS]
                return {'phenotypes': phenotypes}

        elif self.config.name == 'mortality':
            def parse_fn(data):
                return {'mortality': data['label']}

        elif self.config.name == 'length_of_stay':
            def parse_fn(data):
                return {'los': data['label']}
        else:
            assert self.config.name == 'multitask'

            def parse_fn(data):
                return {
                }

        return parse_fn

    def _generate_examples(self, filepath):
        """ Yields examples. """
        # TODO: This method will receive as arguments the `gen_kwargs` defined in the previous `_split_generators` method.
        # It is in charge of opening the given file and yielding (key, example) tuples from the dataset
        # The key is not important, it's more here for legacy reason (legacy from tfds)
        allowed_features = ['TEXT','label']
        data_df = pd.read_csv(filepath)
        data_df = data_df[[col for col in data_df.columns if col in allowed_features]]
        label_parse_fn = self._get_label_parse_fn()
        for id_ in range(len(data_df)):
            row = data_df.iloc[id_]
            ex = {'text':row['TEXT'],**label_parse_fn(row)}
            yield id_, ex
        
#         with open(filepath, encoding="utf-8") as f:
#             r = csv.DictReader(f, quoting=csv.QUOTE_NONE)
#             label_parse_fn = self._get_label_parse_fn()
#             for id_, row in enumerate(r):
#                 yield id_, {
#                     'hadm_id': row['hadm_id'],
#                     'timestamp': date_parser.parse(row['timestamp']).timestamp(),
#                     'text': row['observations'],
#                     **label_parse_fn(row)
#                 }

print('Imported Data Loader')