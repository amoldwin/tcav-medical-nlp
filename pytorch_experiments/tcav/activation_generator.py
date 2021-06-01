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
# from multiprocessing import dummy as multiprocessing
import os
import os.path
import numpy as np
import PIL.Image
import six
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import LongformerTokenizer
import pandas as pd
import time


tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

from transformers import BertTokenizer, DistilBertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def cnn_encode(text_ex):
    with open('resources/tokenizer_final.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    return pad_sequences(tokenizer.texts_to_sequences([text_ex]), padding='post', maxlen=4096)[0]

def bert_encode(data):
    encoded =  bert_tokenizer.encode_plus(
            text=data,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            return_attention_mask=True,      # Return attention mask
            return_token_type_ids = True,
            truncation=True
            )
    encoded=encoded['input_ids']
    encoded = encoded 
    encoded = encoded + [bert_tokenizer.pad_token_id]*(512-len(encoded))
    return np.asarray(encoded)


def distilbert_encode(data):
    encoded =  distilbert_tokenizer.encode_plus(
            text=data,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            return_attention_mask=True,      # Return attention mask
            return_token_type_ids = True,
            truncation=True
            )
    encoded=encoded['input_ids']
    encoded = encoded 
    encoded = encoded + [bert_tokenizer.pad_token_id]*(512-len(encoded))
    return np.asarray(encoded)

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

  def get_activations_for_concept(self, concept, bottleneck):
    start = time.time()
    examples = self.get_examples_for_concept(concept)
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
    num_workers =4
    if False:
      pool = multiprocessing.Pool(num_workers)
      output = pool.map(
          lambda i: self.model.run_examples(imgs[i * bs:(i + 1) * bs], bottleneck),
          np.arange(int(imgs.shape[0] / bs) + 1))
    else:
      output = []
      for i in range(int(imgs.shape[0] / bs) + 1):
        output.append(
            self.model.run_examples(imgs[i * bs:(i + 1) * bs], bottleneck))
    output = np.concatenate(output, 0)
    if False and channel_mean and len(output.shape) > 3:
      output = np.mean(output, (1, 2))
    # else:
    #   output = np.reshape(output, [output.shape[0], -1])
    print('_patch_activations time',time.time()-start)
    return output

  def get_activations_for_examples(self, examples, bottleneck):
    return self._patch_activations(imgs=examples, bottleneck=bottleneck,bs=8 )
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

def longformer_encode(sent, maxlen=4096):
# using solution from https://discuss.huggingface.co/t/how-to-truncate-from-the-head-in-autotokenizer/676/3
    encoded_sent = tokenizer.encode_plus(
            text=sent,  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
#             max_length=maxlen,                  # Max length to truncate/pad
#             pad_to_max_length=True,         # Pad sentence to max length
            #return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,      # Return attention mask
            return_token_type_ids = True
            )
    ids = encoded_sent['input_ids']
    if len(ids)>maxlen:
        ids = [ids[0]] + ids[-1*(maxlen-1):] 
    else:
        ids = ids + ([1] * (maxlen-len(ids)))
    encoded_sent['input_ids']= ids
        
    ids = encoded_sent['token_type_ids']
    if len(ids)>=maxlen:
        ids = [ids[0]] + ids[-1*(maxlen-1):] 
    else:
        ids = ids + ([0] * (maxlen-len(ids)))

    encoded_sent['token_type_ids']=ids
    
    ids = encoded_sent['attention_mask']
    if len(ids)>maxlen:
        ids = [ids[0]] + ids[-1*(maxlen-1):] 
    else:
        ids = ids + ([0] * (maxlen-len(ids)))
    encoded_sent['attention_mask']=ids

#     print(encoded_sent)
#     print(tokenizer.convert_ids_to_tokens(encoded_sent['input_ids']))
    return encoded_sent


class ImageActivationGenerator(ActivationGeneratorBase):
  """Activation generator for a basic image model"""

  def __init__(self,
               model,
               source_dir,
               acts_dir,
               max_examples=10,
               normalize_image=True,
              document_encoder=None):
    """Initialize ImageActivationGenerator class."

    Args:
      normalize_image: A boolean indicating whether image pixels should be
        normalized to between 0 and 1.
    """
    self.source_dir = source_dir
    self.normalize_image = normalize_image
    self.document_encoder=document_encoder
    super(ImageActivationGenerator, self).__init__(model, acts_dir,
                                                   max_examples)

  def get_examples_for_concept(self, concept):
    start=time.time()
    encode_function=self.document_encoder
    concept_dir = os.path.join(self.source_dir, 'concept_listfiles')
    # print(concept)
    print(concept, flush=True)
    # time.sleep(.2) 
    # img_paths = [
    #     os.path.join(concept_dir, d) for d in tf.io.gfile.listdir(concept_dir)
    # ]
    # imgs = self.load_images_from_files(
    #     img_paths, self.max_examples, shape=self.model.get_image_shape(), encode_function=encode_function)
    encoded_name = concept+'_encoded_'+encode_function.__name__
    if encoded_name not in os.listdir(concept_dir):
      print('not present')
      imgs_df = pd.read_csv(concept_dir+'/'+concept)
      imgs =[encode_function(x) for x in list(imgs_df['TEXT'])]
      pd.DataFrame(imgs).to_csv(concept_dir+'/'+encoded_name, index=False)
    else:
      print('present')
    imgs =  pd.read_csv(concept_dir+'/'+encoded_name)
    #to use a random sampling, change 'head' to 'sample'
    imgs = imgs.head(min(len(imgs),self.max_examples))
    imgs = list(imgs.values )
    print('get_examples_for_concept time',time.time()-start)
    return imgs


  # def load_image_from_file(self, filename, shape, encode_function):
  #   """Given a filename, try to open the file.

  #   If failed, return None.

  #   Args:
  #     filename: location of the image file
  #     shape: the shape of the image file to be scaled

  #   Returns:
  #     the image if succeeds, None if fails.

  #   Rasies:
  #     exception if the image was not the right shape.
  #   """
  #   if not tf.io.gfile.exists(filename):
  #     tf.compat.v1.logging.error('Cannot find file: {}'.format(filename))
  #     return None
  #   if not filename.endswith('txt'):
  #     img = ''
  #   else:
  #       img = open(filename, 'r', encoding="utf8", errors='ignore').read()
  #   encoded = encode_function(img)
  #   return encoded

#     except Exception as e:
#       tf.compat.v1.logging.info(e)
#       return None
#     return img

  # def load_images_from_files(self,
  #                            filenames,
  #                            max_imgs=500,
  #                            do_shuffle=True,
  #                            run_parallel=False,
  #                            shape=(4096,),
  #                            num_workers=50, encode_function=None):
  #   """Return image arrays from filenames.

  #   Args:
  #     filenames: locations of image files.
  #     max_imgs: maximum number of images from filenames.
  #     do_shuffle: before getting max_imgs files, shuffle the names or not
  #     run_parallel: get images in parallel or not
  #     shape: desired shape of the image
  #     num_workers: number of workers in parallelization.

  #   Returns:
  #     image arrays

  #   """
  #   imgs = []
  #   # First shuffle a copy of the filenames.
  #   filenames = filenames[:]
  #   if do_shuffle:
  #     np.random.shuffle(filenames)

  #   if run_parallel:
  #     pool = multiprocessing.Pool(num_workers)
    
  #     imgs = pool.map(
  #         lambda filename: self.load_image_from_file(filename, shape, encode_function=encode_function),
  #         filenames[:max_imgs])
        
        
  #     imgs = [img for img in imgs if img is not None]
  #     if len(imgs) <= 1:
  #       raise ValueError(
  #           'You must have more than 1 image in each class to run TCAV.')
  #   else:
  #     for filename in filenames:
  #       img = self.load_image_from_file(filename, shape, encode_function=encode_function)
  #       if img is not None:
  #         imgs.append(img)
  #       if len(imgs) >= max_imgs:
  #         break
  #     if len(imgs) <= 1:
  #       raise ValueError(
  #           'You must have more than 1 image in each class to run TCAV.')

  #   return np.array(imgs)


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
