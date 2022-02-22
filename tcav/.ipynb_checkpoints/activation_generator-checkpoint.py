"""Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import ABCMeta
from abc import abstractmethod
import os
import os.path
import numpy as np
import PIL.Image
import six
import tensorflow as tf
import pickle

import pandas as pd
import time

from encode_functions import *

dirs_dct = dict(list(pd.read_csv('../directory_paths.csv')['paths'].apply(eval)))


class ActivationGeneratorInterface(six.with_metaclass(ABCMeta, object)):
    """Interface for an activation generator for a model"""

    @abstractmethod
    def process_and_load_activations(self, bottleneck_names, concepts):
        pass

    @abstractmethod
    def get_model(self):
        pass


class ActivationGeneratorBase(ActivationGeneratorInterface):
    """Basic abstract activation generator for a model"""

    def __init__(self, model, acts_dir, max_examples=500):
        self.model = model
        self.acts_dir = acts_dir
        self.max_examples = max_examples

    def get_model(self):
        return self.model

    @abstractmethod
    def get_examples_for_concept(self, concept):
        pass

    def get_activations_for_concept(self, concept, bottleneck, shuffled=True, csv_dir=None):
        #if all activations have been saved to dataframes in advance along with their rowids, we can simply find them
        if self.from_row_id:
          if concept.startswith('allnegative') and not (concept in os.listdir(self.concept_dir)):
            filename = '_'.join(concept.split('_')[:-1])
            ids_to_get = pd.read_csv(self.concept_dir+'/'+filename)[['ROW_ID']].sample(self.max_examples)
            ids_to_get.to_csv(os.path.join(self.concept_dir,concept))
          else:  
            ids_to_get = pd.read_csv(self.concept_dir+'/'+concept)[['ROW_ID']]
          chunks = []
          for fn in os.listdir(self.all_acts_dir):
            fn_path = os.path.join(self.all_acts_dir,fn)
            chunk_acts = ids_to_get.merge(pd.read_pickle(fn_path), on='ROW_ID', how='inner' )
            chunks.append(chunk_acts)
          all_concept_acts_df = pd.concat(chunks, ignore_index=True)
          print('getting '+'concept '+'from row_ids',flush=True)
          return np.array(all_concept_acts_df['activations'].values.tolist())
        start = time.time()

        examples = self.get_examples_for_concept(concept, shuffled=shuffled,csv_dir=csv_dir)

        return self.get_activations_for_examples(examples, bottleneck)
        print('get_activations_for_concept time',time.time()-start)

    def _patch_activations(self, imgs, bottleneck, bs=8, channel_mean=None):
        start = time.time()
        """Returns activations of a list of imgs.
    Args:
      imgs: List/array of images to calculate the activations of
      bottleneck: Name of the bottleneck layer of the model where activations
        are calculated
      bs: The batch size for calculating activations. (To control computational
        cost)
      channel_mean: If true, the activations are averaged across channel.
    Returns:
      The array of activations
    """
        # if channel_mean is None:
        #   channel_mean = self.channel_mean
        imgs = np.asarray(imgs)
        num_workers = 0

        output = []
        for i in range(int(np.ceil(imgs.shape[0] / bs))):
            output.append(
                self.model.run_examples(imgs[i * bs:(i + 1) * bs], bottleneck))

        output = np.concatenate(output, 0)

        print('_patch_activations time', time.time() - start)
        return output

    def get_activations_for_examples(self, examples, bottleneck):
        bs = 8
        if self.document_encoder.__name__ == 'logistic_bert_encode':
            bs = 64
        return self._patch_activations(imgs=examples, bottleneck=bottleneck, bs=bs)
        acts = self.model.run_examples(examples, bottleneck)
        return self.model.reshape_activations(acts).squeeze()

    def process_and_load_activations(self, bottleneck_names, concepts):
        acts = {}
        if self.acts_dir and not tf.io.gfile.exists(self.acts_dir):
            tf.io.gfile.makedirs(self.acts_dir)

        for concept in concepts:
            if concept not in acts:
                acts[concept] = {}
            for bottleneck_name in bottleneck_names:
                acts_path = os.path.join(self.acts_dir, 'acts_{}_{}'.format(
                    concept, bottleneck_name)) if self.acts_dir else None
                if acts_path and tf.io.gfile.exists(acts_path):
                    with tf.io.gfile.GFile(acts_path, 'rb') as f:
                        acts[concept][bottleneck_name] = np.load(
                            f, allow_pickle=True).squeeze()
                        tf.compat.v1.logging.info('Loaded {} shape {}'.format(
                            acts_path, acts[concept][bottleneck_name].shape))
                else:
                    acts[concept][bottleneck_name] = self.get_activations_for_concept(
                        concept, bottleneck_name)
                    if acts_path:
                        tf.compat.v1.logging.info(
                            '{} does not exist, Making one...'.format(acts_path))
                        tf.io.gfile.mkdir(os.path.dirname(acts_path))
                        with tf.io.gfile.GFile(acts_path, 'w') as f:
                            np.save(f, acts[concept][bottleneck_name], allow_pickle=False)
        return acts




class ImageActivationGenerator(ActivationGeneratorBase):
    """Activation generator for a basic image model"""

    def __init__(self,
                 model,
                 source_dir,
                 acts_dir,
                 max_examples=10,
                 normalize_image=True,
                 document_encoder=None,
                 concept_dir=dirs_dct['concepts_dir'],
                 from_row_id=False,
                 all_acts_dir=None):
        """Initialize ImageActivationGenerator class."

    Args:
      normalize_image: A boolean indicating whether image pixels should be
        normalized to between 0 and 1.
    """
        self.source_dir = source_dir
        self.concept_dir = concept_dir
        self.normalize_image = normalize_image
        self.document_encoder = document_encoder
        self.from_row_id = from_row_id
        self.all_acts_dir = all_acts_dir
        super(ImageActivationGenerator, self).__init__(model, acts_dir,
                                                       max_examples)

    def get_examples_for_concept(self, concept, shuffled=True, csv_dir=None):
        start = time.time()
        encode_function = self.document_encoder
        concept_dir = self.concept_dir
        if not csv_dir == None:
            concept_dir = csv_dir
        print(concept, flush=True)
        encoded_path = os.path.join(concept_dir,'encoded_concepts', self.document_encoder.__name__, concept)
        if not tf.io.gfile.exists(encoded_path):
            print('preencoded version not present, tokenizing now...')
            imgs_df = pd.read_csv(concept_dir + '/' + concept)
            imgs = [encode_function(x) for x in list(imgs_df['TEXT'])]
            if 'rhn' in encode_function.__name__:
                df = pd.DataFrame()
                df['input_ids'] = pd.Series(imgs).apply(lambda x: list(x))
                measurement_types = ['mean', 'count', 'min', 'max', 'dx', 'MED_', 'median', 'stdev', 'AGE', 'GENDER']
                df['structured_features'] = imgs_df[
                    [x for x in imgs_df.columns if any([y in x for y in measurement_types])]].apply(
                    lambda row: [float(x) for x in list(row)], axis=1)
                df.to_csv(encoded_path, index=False)
                del df
            else:
                pd.DataFrame(imgs).to_csv(encoded_path, index=False)
        else:
            print('loading tokenized examples')
        imgs = pd.read_csv(encoded_path)
        # to use a random sampling, change 'head' to 'sample'
        if shuffled:
            imgs = imgs.sample(min(len(imgs), self.max_examples))
        else:
            imgs = imgs.head(min(len(imgs), self.max_examples))
        if 'rhn' in encode_function.__name__:
            imgs['structured_features'] = imgs['structured_features'].apply(eval)
            return imgs
        imgs = list(imgs.values)
        print('get_examples_for_concept time', time.time() - start)
        return imgs



"""Discrete activation generators"""


class DiscreteActivationGeneratorBase(ActivationGeneratorBase):
    """ Base class for discrete data. """

    def __init__(self, model, source_dir, acts_dir, max_examples):
        self.source_dir = source_dir
        super(DiscreteActivationGeneratorBase, self).__init__(
            model=model, acts_dir=acts_dir, max_examples=max_examples)

    def get_examples_for_concept(self, concept):
        """Extracts examples for a concept and transforms them to the desired format.

    Args:
      concept: Name of a concept. Names for the folder containing data for that
        concept. Path is fetched based on the source_dir used by the activation
        generator

    Returns:
          data_parsed:
            Examples from the data folder. Format is according to the
            load_data() and transform_data() functions.

    """
        data = self.load_data(concept)
        data_parsed = self.transform_data(data)
        return data_parsed

    @abstractmethod
    def load_data(self, concept):
        """Extracts data from a source and returns it in a user specified format.

    It takes in as input the name for a concept folder that lies inside of the
    source_dir.

    Args
      concept: name of the concept (e.g., name of a folder that contains concept
      examples in source_dir). They should be located in source_dir.
    Returns:
      data parsed
    """
        # Needs to be implemented
        raise NotImplementedError()

    def transform_data(self, data):
        """Transforms data into a format that can be directly processed by the model.

    Once the data is parsed, use this function to transform the data into the
    format your model needs. Some example transformations can be:
      - Converting to proto type
      - Encoding categorical features
      - Tokenizing an input sequence

    Args:
      data: The previously extracted data from load_data

    Returns:
      Transformed data, in the desired dormat
    """
        # By default, regurgitates input data if not implemented
        return data
