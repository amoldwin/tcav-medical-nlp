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
from six.moves import zip
import numpy as np
import six
import tensorflow as tf
from google.protobuf import text_format
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

class ModelWrapper(six.with_metaclass(ABCMeta, object)):
  """Simple wrapper of the for models with session object for TCAV.
    Supports easy inference with no need to deal with the feed_dicts.
  """

  @abstractmethod
  def __init__(self, model_path=None, node_dict=None):
    """Initialize the wrapper.
    Optionally create a session, load
    the model from model_path to this session, and map the
    input/output and bottleneck tensors.
    Args:
      model_path: one of the following: 1) Directory path to checkpoint 2)
        Directory path to SavedModel 3) File path to frozen graph.pb 4) File
        path to frozen graph.pbtxt
      node_dict: mapping from a short name to full input/output and bottleneck
        tensor names. Users should pass 'input' and 'prediction'
        as keys and the corresponding input and prediction tensor
        names as values in node_dict. Users can additionally pass bottleneck
        tensor names for which gradient Ops will be added later.
    """
    # A dictionary of bottleneck tensors.
    self.bottlenecks_tensors = None
    # A dictionary of input, 'logit' and prediction tensors.
    self.ends = None
    # The model name string.
    self.model_name = None
    # a place holder for index of the neuron/class of interest.
    # usually defined under the graph. For example:
    # with g.as_default():
    #   self.tf.placeholder(tf.int64, shape=[None])
    self.y_input = None
    # The tensor representing the loss (used to calculate derivative).
    self.loss = None
    # If tensors in the loaded graph are prefixed with 'import/'
    self.import_prefix = False

    if model_path:
      self._try_loading_model(model_path)
    if node_dict:
      self._find_ends_and_bottleneck_tensors(node_dict)

  def _try_loading_model(self, model_path):
    """ Load model from model_path.
    TF models are often saved in one of the three major formats:
      1) Checkpoints with ckpt.meta, ckpt.data, and ckpt.index.
      2) SavedModel format with saved_model.pb and variables/.
      3) Frozen graph in .pb or .pbtxt format.
    When model_path is specified, model is loaded in one of the
    three formats depending on the model_path. When model_path is
    ommitted, child wrapper is responsible for loading the model.
    """
    try:
      self.sess = tf.compat.v1.Session(graph=tf.Graph())
      with self.sess.graph.as_default():
        if tf.io.gfile.isdir(model_path):
          ckpt = tf.train.latest_checkpoint(model_path)
          if ckpt:
            tf.compat.v1.logging.info('Loading from the latest checkpoint.')
            saver = tf.compat.v1.train.import_meta_graph(ckpt + '.meta')
            saver.restore(self.sess, ckpt)
          else:
            tf.compat.v1.logging.info('Loading from SavedModel dir.')
            tf.compat.v1.saved_model.loader.load(self.sess, ['serve'], model_path)
        else:
          input_graph_def = tf.compat.v1.GraphDef()
          if model_path.endswith('.pb'):
            tf.compat.v1.logging.info('Loading from frozen binary graph.')
            with tf.io.gfile.GFile(model_path, 'rb') as f:
              input_graph_def.ParseFromString(f.read())
          else:
            tf.compat.v1.logging.info('Loading from frozen text graph.')
            with tf.io.gfile.GFile(model_path) as f:
              text_format.Parse(f.read(), input_graph_def)
          tf.import_graph_def(input_graph_def)
          self.import_prefix = True

    except Exception as e:
      template = 'An exception of type {0} occurred ' \
                 'when trying to load model from {1}. ' \
                 'Arguments:\n{2!r}'
      tf.compat.v1.logging.warn(template.format(type(e).__name__, model_path, e.args))

  def _find_ends_and_bottleneck_tensors(self, node_dict):
    """ Find tensors from the graph by their names.
    Depending on how the model is loaded, tensors in the graph
    may or may not have 'import/' prefix added to every tensor name.
    This is true even if the tensors already have 'import/' prefix.
    The 'ends' and 'bottlenecks_tensors' dictionary should map to tensors
    with the according name.
    """
    self.bottlenecks_tensors = {}
    self.ends = {}
    for k, v in six.iteritems(node_dict):
      if self.import_prefix:
        v = 'import/' + v
      tensor = self.sess.graph.get_operation_by_name(v.strip(':0')).outputs[0]
      if k == 'input' or k == 'prediction':
        self.ends[k] = tensor
      else:
        self.bottlenecks_tensors[k] = tensor

  def _make_gradient_tensors(self):
    """Makes gradient tensors for all bottleneck tensors."""

    self.bottlenecks_gradients = {}
    for bn in self.bottlenecks_tensors:
      self.bottlenecks_gradients[bn] = tf.gradients(
          ys=self.loss, xs=self.bottlenecks_tensors[bn])[0]

  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """


    # print('bn name',bottleneck_name)
    # print('gradient',self.bottlenecks_gradients[bottleneck_name])
    # print('acts', acts)
    # print('y',y)
    # print(self.y_input)
    # print(self.id_)
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: y,
        self.id_: [example]
    })

  def get_predictions(self, examples):
    """Get prediction of the examples.
    Args:
      imgs: array of examples to get predictions
    Returns:
      array of predictions
    """
    return self.adjust_prediction(
        self.sess.run(self.ends['prediction'], {self.ends['input']: examples}))

  def adjust_prediction(self, pred_t):
    """Adjust the prediction tensor to be the expected shape.
    Defaults to a no-op, but necessary to override for GoogleNet
    Returns:
      pred_t: pred_tensor.
    """
    return pred_t

  def reshape_activations(self, layer_acts):
    """Reshapes layer activations as needed to feed through the model network.
    Override this for models that require reshaping of the activations for use
    in TCAV.
    Args:
      layer_acts: Activations as returned by run_examples.
    Returns:
      Activations in model-dependent form; the default is a squeezed array (i.e.
      at most one dimensions of size 1).
    """
    return np.asarray(layer_acts).squeeze()

  def label_to_id(self, label):
    """Convert label (string) to index in the logit layer (id).
    Override this method if label to id mapping is known. Otherwise,
    default id 0 is used.
    """
    tf.compat.v1.logging.warn('label_to_id undefined. Defaults to returning 0.')
    return 0


  def id_to_label(self, idx):
    """Convert index in the logit layer (id) to label (string).
    Override this method if id to label mapping is known.
    """
    return str(idx)

  def run_examples(self, examples, bottleneck_name):
    """Get activations at a bottleneck for provided examples.
    Args:
      examples: example data to feed into network.
      bottleneck_name: string, should be key of self.bottlenecks_tensors
    Returns:
      Activations in the given layer.
    """
    return self.sess.run(self.bottlenecks_tensors[bottleneck_name],
                         {self.ends['input']: examples})


class ImageModelWrapper(ModelWrapper):
  """Wrapper base class for image models."""

  def __init__(self, image_shape):
    super(ModelWrapper, self).__init__()
    # shape of the input image in this model
    self.image_shape = image_shape

  def get_image_shape(self):
    """returns the shape of an input image."""
    return self.image_shape


class PublicImageModelWrapper(ImageModelWrapper):
  """Simple wrapper of the public image models with session object."""

  def __init__(self, sess, model_fn_path, labels_path, image_shape,
               endpoints_dict, scope):
    super(PublicImageModelWrapper, self).__init__(image_shape)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()
    self.ends = PublicImageModelWrapper.import_graph(
        model_fn_path, endpoints_dict, self.image_value_range, scope=scope)
    self.bottlenecks_tensors = PublicImageModelWrapper.get_bottleneck_tensors(
        scope)
    graph = tf.compat.v1.get_default_graph()

    # Construct gradient ops.
    with graph.as_default():
      self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None])

      self.pred = tf.expand_dims(self.ends['prediction'][0], 0)
      self.loss = tf.reduce_mean(
          input_tensor=tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(
              labels=tf.one_hot(
                  self.y_input,
                  self.ends['prediction'].get_shape().as_list()[1]),
              logits=self.pred))
    self._make_gradient_tensors()

  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)

  @staticmethod
  def create_input(t_input, image_value_range):
    """Create input tensor."""
    def forget_xy(t):
      """Forget sizes of dimensions [1, 2] of a 4d tensor."""
      zero = tf.identity(0)
      return t[:, zero:, zero:, :]

    t_prep_input = t_input
    if len(t_prep_input.shape) == 3:
      t_prep_input = tf.expand_dims(t_prep_input, 0)
    t_prep_input = forget_xy(t_prep_input)
    lo, hi = image_value_range
    t_prep_input = lo + t_prep_input * (hi - lo)
    return t_input, t_prep_input


  # From Alex's code.
  @staticmethod
  def get_bottleneck_tensors(scope):
    """Add Inception bottlenecks and their pre-Relu versions to endpoints dict."""
    graph = tf.compat.v1.get_default_graph()
    bn_endpoints = {}
    for op in graph.get_operations():
      if op.name.startswith(scope + '/') and 'Concat' in op.type:
        name = op.name.split('/')[1]
        bn_endpoints[name] = op.outputs[0]
    return bn_endpoints

  # Load graph and import into graph used by our session
  @staticmethod
  def import_graph(saved_path, endpoints, image_value_range, scope='import'):
    t_input = tf.compat.v1.placeholder(np.float32, [None, None, None, 3])
    graph = tf.Graph()
    assert graph.unique_name(scope, False) == scope, (
        'Scope "%s" already exists. Provide explicit scope names when '
        'importing multiple instances of the model.') % scope

    graph_def = tf.compat.v1.GraphDef.FromString(
        tf.io.gfile.GFile(saved_path, 'rb').read())

    with tf.compat.v1.name_scope(scope) as sc:
      t_input, t_prep_input = PublicImageModelWrapper.create_input(
          t_input, image_value_range)

      graph_inputs = {}
      graph_inputs[endpoints['input']] = t_prep_input
      myendpoints = tf.import_graph_def(
          graph_def, graph_inputs, list(endpoints.values()), name=sc)
      myendpoints = dict(list(zip(list(endpoints.keys()), myendpoints)))
      myendpoints['input'] = t_input
    return myendpoints


class GoogleNetWrapper_public(PublicImageModelWrapper):

  def __init__(self, sess, model_saved_path, labels_path):
    image_shape_v1 = [224, 224, 3]
    self.image_value_range = (-117, 255 - 117)
    endpoints_v1 = dict(
        input='input:0',
        logit='softmax2_pre_activation:0',
        prediction='output2:0',
        pre_avgpool='mixed5b:0',
        logit_weight='softmax2_w:0',
        logit_bias='softmax2_b:0',
    )
    self.sess = sess
    super(GoogleNetWrapper_public, self).__init__(
        sess,
        model_saved_path,
        labels_path,
        image_shape_v1,
        endpoints_v1,
        scope='v1')
    self.model_name = 'GoogleNet_public'

  def adjust_prediction(self, pred_t):
    # Each pred outputs 16, 1008 matrix. The prediction value is the first row.
    # Following tfzoo convention.
    return pred_t[::16]

# class LongformerWrapper_public(PublicImageModelWrapper):

#   def __init__(self, sess, model_saved_path, labels_path):
#     input_shape = [4096,3]
#     #inputs range from  0 to tokenizer.vocab_size
#     self.image_value_range = (0, 50264)
# #     ['input_4',
# #  'input_5',
# #  'input_6',
# #  'tf_longformer_model_1',
# #  'global_average_pooling1d_1',
# #  'dropout_99',
# #  'dense']
#     endpoints_v1 = dict(
#         input='input:0',
#         logit='softmax2_pre_activation:0',
#         prediction='output2:0',
#         pre_avgpool='mixed5b:0',
#         logit_weight='softmax2_w:0',
#         logit_bias='softmax2_b:0',
#     )
#     self.sess = sess
#     super(GoogleNetWrapper_public, self).__init__(
#         sess,
#         model_saved_path,
#         labels_path,
#         image_shape_v1,
#         endpoints_v1,
#         scope='v1')
#     self.model_name = 'GoogleNet_public'

#   def adjust_prediction(self, pred_t):
#     # Each pred outputs 16, 1008 matrix. The prediction value is the first row.
#     # Following tfzoo convention.
#     return pred_t[::16]

class InceptionV3Wrapper_public(PublicImageModelWrapper):
  def __init__(self, sess, model_saved_path, labels_path):
    self.image_value_range = (-1, 1)
    image_shape_v3 = [299, 299, 3]
    endpoints_v3 = dict(
        input='Mul:0',
        logit='softmax/logits:0',
        prediction='softmax:0',
        pre_avgpool='mixed_10/join:0',
        logit_weight='softmax/weights:0',
        logit_bias='softmax/biases:0',
    )

    self.sess = sess
    super(InceptionV3Wrapper_public, self).__init__(
        sess,
        model_saved_path,
        labels_path,
        image_shape_v3,
        endpoints_v3,
        scope='v3')
    self.model_name = 'InceptionV3_public'


class MobilenetV2Wrapper_public(PublicImageModelWrapper):

  def __init__(self, sess, model_saved_path, labels_path):
    self.image_value_range = (-1, 1)
    image_shape_v2 = [224, 224, 3]
    endpoints_v2 = dict(
        input='input:0',
        prediction='MobilenetV2/Predictions/Reshape:0',
    )

    self.sess = sess
    super(MobilenetV2Wrapper_public, self).__init__(
        sess,
        model_saved_path,
        labels_path,
        image_shape_v2,
        endpoints_v2,
        scope='MobilenetV2')

    # define bottleneck tensors and their gradients
    self.bottlenecks_tensors = self.get_bottleneck_tensors_mobilenet(
        scope='MobilenetV2')
    # Construct gradient ops.
    g = tf.compat.v1.get_default_graph()
    self._make_gradient_tensors()
    self.model_name = 'MobilenetV2_public'

  @staticmethod
  def get_bottleneck_tensors_mobilenet(scope):
    """Add Inception bottlenecks and their pre-Relu versions to endpoints dict."""
    graph = tf.compat.v1.get_default_graph()
    bn_endpoints = {}
    for op in graph.get_operations():
      if 'add' in op.name and 'gradients' not in op.name and 'add' == op.name.split(
          '/')[-1]:
        name = op.name.split('/')[-2]
        bn_endpoints[name] = op.outputs[0]
    return bn_endpoints


class KerasModelWrapper(ModelWrapper):
  """ ModelWrapper for keras models
    By default, assumes that your model contains one input node, one output head
    and one loss function.
    Computes gradients of the output layer in respect to a CAV.
    Args:
        sess: Tensorflow session we will use for TCAV.
        model_path: Path to your model.h5 file, containing a saved trained
          model.
        labels_path: Path to a file containing the labels for your problem. It
          requires a .txt file, where every line contains a label for your
          model. You want to make sure that the order of labels in this file
          matches with the logits layers for your model, such that file[i] ==
          model_logits[i]
  """

  def __init__( 
      self,
      sess,
      model_path,
      labels_path  ):
    self.sess = sess
    super(KerasModelWrapper, self).__init__()
    self.import_keras_model(model_path)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()
    self.image_shape = [4096]
    # Construct gradient ops. Defaults to using the model's output layer
    self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None])
    self.loss = self.model.loss_functions[0](self.y_input,
                                             self.model.outputs[0])
    self._make_gradient_tensors()
  def get_image_shape(self):
    """returns the shape of an input image."""
    return self.image_shape
  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)

  def import_keras_model(self, saved_path):
    """Loads keras model, fetching bottlenecks, inputs and outputs."""
    self.ends = {}
    self.model = tf.keras.models.load_model(saved_path)
    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()
    
  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: y
    })

  def get_bottleneck_tensors(self):
    self.bottlenecks_tensors = {}
    layers = self.model.layers
    for layer in layers:
      if 'input' not in layer.name:
        self.bottlenecks_tensors[layer.name] = layer.output
    return[layer.name for layer in layers]

  def get_inputs_and_outputs_and_ends(self):
    self.ends['input'] = self.model.inputs[0]
    self.ends['prediction'] = self.model.outputs[0]

class LongformerWrapper(ModelWrapper):
  """ ModelWrapper for keras models
    By default, assumes that your model contains one input node, one output head
    and one loss function.
    Computes gradients of the output layer in respect to a CAV.
    Args:
        sess: Tensorflow session we will use for TCAV.
        model_path: Path to your model.h5 file, containing a saved trained
          model.
        labels_path: Path to a file containing the labels for your problem. It
          requires a .txt file, where every line contains a label for your
          model. You want to make sure that the order of labels in this file
          matches with the logits layers for your model, such that file[i] ==
          model_logits[i]
  """

  def __init__( 
      self,
      sess,
      model_path,
      labels_path  ):
    self.sess = sess
    super(LongformerWrapper, self).__init__()
    self.import_bert_model(model_path)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()
    self.image_shape = [4096]
    # Construct gradient ops. Defaults to using the model's output layer
    self.y_input = tf.compat.v1.placeholder(tf.int64, shape=[None])
    self.loss = self.model.loss_functions[0](self.y_input,
                                             self.model.outputs[0])
    self._make_gradient_tensors()
  def get_image_shape(self):
    """returns the shape of an input image."""
    return self.image_shape
  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)
  def import_bert_model(self, saved_path):
    from tensorflow.keras import backend as K
    import numpy as np
    import os
    import random
    from transformers import TFAutoModel, AutoTokenizer, LongformerConfig, TFLongformerModel, LongformerTokenizer
    self.ends = {}
    opt = Adam(learning_rate=2e-5)
    
    id_ = Input((4096,), dtype=tf.int32)
    mask_ = Input((4096,), dtype=tf.int32)
    atn_ = Input((4096,), dtype=tf.int32)
    
    config = LongformerConfig.from_pretrained('allenai/longformer-base-4096')
    config.output_hidden_states = False # Set to True to obtain hidden states
    bert_model = TFLongformerModel.from_pretrained('allenai/longformer-base-4096', config=config)
    bert_model.load_weights(saved_path)
    embedding = bert_model(id_, attention_mask=mask_, token_type_ids=atn_)[0]
    
    x = GlobalAveragePooling1D()(embedding)    
    x = Dropout(0.2)(x)
    out = Dense(1, activation='sigmoid')(x)

    self.model = Model(inputs=[id_, mask_, atn_], outputs=out)
    self.model.compile(loss='binary_crossentropy', optimizer=opt, metrics = [auc])

    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()

  def import_keras_model(self, saved_path):
    """Loads keras model, fetching bottlenecks, inputs and outputs."""
    

    self.model = simple_bert()

    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()

  def get_bottleneck_tensors(self):
    self.bottlenecks_tensors = {}
    layers = self.model.layers
    for layer in layers:
      if 'input' not in layer.name:
        self.bottlenecks_tensors[layer.name] = layer.output
    return[layer.name for layer in layers]

  def get_inputs_and_outputs_and_ends(self):
    self.ends['input'] = self.model.inputs[0]
    self.ends['prediction'] = self.model.outputs[0]


class BertWrapper(ModelWrapper):
  """ ModelWrapper for BERT 
  """
  def __init__( 
      self,
      sess,
      model_path,
      labels_path  ):
    self.sess = sess
    super(BertWrapper, self).__init__()
    self.import_bert_model(model_path)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()
    self.image_shape = [512]
    # Construct gradient ops. Defaults to using the model's output layer
    # self.y_input = tf.compat.v1.placeholder(tf.int32, shape=[None])
    # self.loss = self.model.loss_functions[0](self.y_input,
                                            #  self.model.outputs[0])
    self._make_gradient_tensors()
  def get_image_shape(self):
    """returns the shape of an input image."""
    return self.image_shape
  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)
  def import_bert_model(self, saved_path):
    from tensorflow.keras import backend as K
    import numpy as np
    import os
    import random
    from transformers import TFAutoModel, AutoTokenizer, BertConfig, TFBertForSequenceClassification, BertTokenizer
    self.ends = {}
    opt = Adam(learning_rate=2e-5)
    id_ = Input((512,), dtype=tf.int32)
    label = Input((1,), dtype=tf.int32)
    self.y_input = label
    self.id_ =  id_
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
       # config = BertConfig.from_pretrained("bertcheckpoints/Bert0.h5",
    #                                 output_hidden_states=True)
    model = TFBertForSequenceClassification.from_pretrained(saved_path, output_hidden_states=True)#"bertcheckpoints/Bert0.h5")

    # embedding = model(id_, attention_mask=mask_, token_type_ids=atn_)[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer)
    self.model = model
    self.out = model(id_, labels = label)
    self.loss = self.out.loss
    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()
  
  def get_bottleneck_tensors(self):
    self.bottlenecks_tensors = {}
    layers = list(self.out.hidden_states)
    for i,layer in enumerate(layers):
      if 'input' not in layer.name:
        self.bottlenecks_tensors[str(i)] = layer
    return list(self.bottlenecks_tensors.keys())

  def get_inputs_and_outputs_and_ends(self):
    self.ends['input'] = self.id_ # Input((512,), dtype=tf.int32)#tf.compat.v1.placeholder(tf.int32, (None,512), name='ins')
    self.ends['prediction'] = self.out.logits #self.model.outputs

  def sequence_to_embedding(self,seq):
    embedding_matrix = self.model.bert.embeddings.word_embeddings
    token_ids = seq
    vocab_size = embedding_matrix.get_shape()[0]
    # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
    token_ids_tensor = tf.constant(token_ids, dtype='int32')
    token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)  
    inputs_embeds = tf.matmul(token_ids_tensor_one_hot,embedding_matrix) 
    return inputs_embeds.eval()

  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """
    # print('bn name',bottleneck_name)
    # print('gradient',self.bottlenecks_gradients[bottleneck_name])
    # print('acts', acts)
    # print('y',y)
    # print(self.y_input)
    # print(self.id_)
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: [y],
        self.id_:[example]
    })

class DistilBertWrapper(ModelWrapper):
  """ ModelWrapper for BERT 
  """
  def __init__( 
      self,
      sess,
      model_path,
      labels_path  ):
    self.sess = sess
    super(DistilBertWrapper, self).__init__()
    self.import_bert_model(model_path)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()
    self.image_shape = [512]
    # Construct gradient ops. Defaults to using the model's output layer
    # self.y_input = tf.compat.v1.placeholder(tf.int32, shape=[None])
    # self.loss = self.model.loss_functions[0](self.y_input,
                                            #  self.model.outputs[0])
    self._make_gradient_tensors()
  def get_image_shape(self):
    """returns the shape of an input image."""
    return self.image_shape
  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)
  def import_bert_model(self, saved_path):
    from tensorflow.keras import backend as K
    import numpy as np
    import os
    import random
    from transformers import DistilBertConfig,  DistilBertTokenizer
    from multiclass_model import TFDistilBertForSequenceClassification
    self.ends = {}
    opt = Adam(learning_rate=2e-5)
    id_ = Input((512,), dtype=tf.int32)
    label = Input((1,), dtype=tf.int32)
    self.y_input = label
    self.id_ =  id_
    self.tokenizer = DistilBertTokenizer.from_pretrained('bert-base-uncased')
       # config = BertConfig.from_pretrained("bertcheckpoints/Bert0.h5",
    #                                 output_hidden_states=True)
    model = TFDistilBertForSequenceClassification.from_pretrained(saved_path, output_hidden_states=True)

    # embedding = model(id_, attention_mask=mask_, token_type_ids=atn_)[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer)
    self.model = model
    self.out = model(id_, labels = label)
    self.loss = self.out.loss
    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()
  
  def get_bottleneck_tensors(self):
    self.bottlenecks_tensors = {}
    layers = list(self.out.hidden_states)
    for i,layer in enumerate(layers):
      if 'input' not in layer.name:
        self.bottlenecks_tensors[str(i)] = layer
    self.bottlenecks_tensors['dense0'] =self.model.pre_classifier_acts
    # self.bottlenecks_tensors['dense1'] =self.model.layers[2].output
    return list(self.bottlenecks_tensors.keys())

  def get_inputs_and_outputs_and_ends(self):
    self.ends['input'] = self.id_ # Input((512,), dtype=tf.int32)#tf.compat.v1.placeholder(tf.int32, (None,512), name='ins')
    self.ends['prediction'] = self.out.logits #self.model.outputs

  def sequence_to_embedding(self,seq):
    embedding_matrix = self.model.bert.embeddings.word_embeddings
    token_ids = seq
    vocab_size = embedding_matrix.get_shape()[0]
    # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
    token_ids_tensor = tf.constant(token_ids, dtype='int32')
    token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)  
    inputs_embeds = tf.matmul(token_ids_tensor_one_hot,embedding_matrix) 
    return inputs_embeds.eval()

  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """
    # print('bn name',bottleneck_name)
    # print('gradient',self.bottlenecks_gradients[bottleneck_name])
    # print('acts', acts)
    # print('y',y)
    # print(self.y_input)
    # print(self.id_)
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: [y],
        self.id_:[example]
    })


  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """
    # print('bn name',bottleneck_name)
    # print('gradient',self.bottlenecks_gradients[bottleneck_name])
    # print('acts', acts)
    # print('y',y)
    # print(self.y_input)
    # print(self.id_)
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: [y],
        self.id_:[example]
    })

class DistilBertMulticlassWrapper(ModelWrapper):
  """ ModelWrapper for BERT 
  """
  def __init__( 
      self,
      sess,
      model_path,
      labels_path, 
      class_number  ):
    self.sess = sess
    super(DistilBertMulticlassWrapper, self).__init__()
    self.class_number = class_number
    self.import_bert_model(model_path)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()
    self.image_shape = [512]

    # Construct gradient ops. Defaults to using the model's output layer
    # self.y_input = tf.compat.v1.placeholder(tf.int32, shape=[None])
    # self.loss = self.model.loss_functions[0](self.y_input,
                                            #  self.model.outputs[0])
    self._make_gradient_tensors()
  def get_image_shape(self):
    """returns the shape of an input image."""
    return self.image_shape
  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)
  def import_bert_model(self, saved_path):
    from tensorflow.keras import backend as K
    import numpy as np
    import os
    import random
    from transformers import TFAutoModel, AutoTokenizer, DistilBertConfig, TFDistilBertForSequenceClassification, DistilBertTokenizer
    from multiclass_model import TFDistilBertForMultilabelClassification
    self.ends = {}
    # opt = Adam(learning_rate=2e-5)
    id_ = Input((512,), dtype=tf.int32)
    label = Input((None,), dtype=tf.int32)
    self.label=label
    self.y_input = label#[:,self.class_number]
    self.id_ =  id_
    self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
       # config = BertConfig.from_pretrained("bertcheckpoints/Bert0.h5",
    #                                 output_hidden_states=True)
    learning_rate = 2e-5

    # embedding = model(id_, attention_mask=mask_, token_type_ids=atn_)[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.BinaryCrossentropy()
    # metric = tf.keras.metrics.CategoricalAccuracy()

    model = TFDistilBertForMultilabelClassification.from_pretrained(saved_path, class_number=self.class_number,output_hidden_states=True)#"bertcheckpoints/Bert0.h5")

    # embedding = model(id_, attention_mask=mask_, token_type_ids=atn_)[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer)
    self.model = model
    self.out = model(id_, labels = label)
    self.loss = self.out.loss
    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()
  
  def get_bottleneck_tensors(self):
    self.bottlenecks_tensors = {}
    layers = list(self.out.hidden_states)
    for i,layer in enumerate(layers):
      if 'input' not in layer.name:
        self.bottlenecks_tensors[str(i)] = layer
    return list(self.bottlenecks_tensors.keys())

  def get_inputs_and_outputs_and_ends(self):
    self.ends['input'] = self.id_ # Input((512,), dtype=tf.int32)#tf.compat.v1.placeholder(tf.int32, (None,512), name='ins')
    self.ends['prediction'] = self.out.logits[:,self.class_number] #self.model.outputs

  def sequence_to_embedding(self,seq):
    embedding_matrix = self.model.bert.embeddings.word_embeddings
    token_ids = seq
    vocab_size = embedding_matrix.get_shape()[0]
    # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
    token_ids_tensor = tf.constant(token_ids, dtype='int32')
    token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)  
    inputs_embeds = tf.matmul(token_ids_tensor_one_hot,embedding_matrix) 
    return inputs_embeds.eval()

  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: [y],
        self.id_:[example]
    })

class BertMulticlassWrapper(ModelWrapper):
  """ ModelWrapper for BERT 
  """
  def __init__( 
      self,
      sess,
      model_path,
      labels_path, 
      class_number  ):
    self.sess = sess
    super(BertMulticlassWrapper, self).__init__()
    self.class_number = class_number
    self.import_bert_model(model_path)
    self.labels = tf.io.gfile.GFile(labels_path).read().splitlines()
    self.image_shape = [512]

    # Construct gradient ops. Defaults to using the model's output layer
    # self.y_input = tf.compat.v1.placeholder(tf.int32, shape=[None])
    # self.loss = self.model.loss_functions[0](self.y_input,
                                            #  self.model.outputs[0])
    self._make_gradient_tensors()
  def get_image_shape(self):
    """returns the shape of an input image."""
    return self.image_shape
  def id_to_label(self, idx):
    return self.labels[idx]

  def label_to_id(self, label):
    return self.labels.index(label)
  def import_bert_model(self, saved_path):
    from tensorflow.keras import backend as K
    import numpy as np
    import os
    import random
    from transformers import TFAutoModel, AutoTokenizer, BertConfig, TFBertForSequenceClassification, BertTokenizer
    from multiclass_model import TFBertForMultilabelClassification
    self.ends = {}
    # opt = Adam(learning_rate=2e-5)
    id_ = Input((512,), dtype=tf.int32)
    label = Input((None,25), dtype=tf.int32)
    self.y_input = label[:,self.class_number]
    self.id_ =  id_
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
       # config = BertConfig.from_pretrained("bertcheckpoints/Bert0.h5",
    #                                 output_hidden_states=True)
    learning_rate = 2e-5

    # embedding = model(id_, attention_mask=mask_, token_type_ids=atn_)[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,epsilon=1e-08, clipnorm=1)
    # we do not have one-hot vectors, we can use sparce categorical cross entropy and accuracy
    loss = tf.keras.losses.BinaryCrossentropy()
    # metric = tf.keras.metrics.CategoricalAccuracy()

    model = TFBertForMultilabelClassification.from_pretrained(saved_path, output_hidden_states=True)#"bertcheckpoints/Bert0.h5")

    # embedding = model(id_, attention_mask=mask_, token_type_ids=atn_)[0]
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(loss=loss, optimizer=optimizer)
    self.model = model
    self.out = model(id_, labels = label)
    self.loss = self.out.loss
    self.get_bottleneck_tensors()
    self.get_inputs_and_outputs_and_ends()
  
  def get_bottleneck_tensors(self):
    self.bottlenecks_tensors = {}
    layers = list(self.out.hidden_states)
    for i,layer in enumerate(layers):
      if 'input' not in layer.name:
        self.bottlenecks_tensors[str(i)] = layer
    return list(self.bottlenecks_tensors.keys())

  def get_inputs_and_outputs_and_ends(self):
    self.ends['input'] = self.id_ # Input((512,), dtype=tf.int32)#tf.compat.v1.placeholder(tf.int32, (None,512), name='ins')
    self.ends['prediction'] = self.out.logits[:,self.class_number] #self.model.outputs
    # return self.ends

  def sequence_to_embedding(self,seq):
    embedding_matrix = self.model.bert.embeddings.word_embeddings
    token_ids = seq
    vocab_size = embedding_matrix.get_shape()[0]
    # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
    token_ids_tensor = tf.constant(token_ids, dtype='int32')
    token_ids_tensor_one_hot = tf.one_hot(token_ids_tensor, vocab_size)  
    inputs_embeds = tf.matmul(token_ids_tensor_one_hot,embedding_matrix) 
    return inputs_embeds.eval()

  def get_gradient(self, acts, y, bottleneck_name, example):
    """Return the gradient of the loss with respect to the bottleneck_name.
    Args:
      acts: activation of the bottleneck
      y: index of the logit layer
      bottleneck_name: name of the bottleneck to get gradient wrt.
      example: input example. Unused by default. Necessary for getting gradients
        from certain models, such as BERT.
    Returns:
      the gradient array.
    """
    return self.sess.run(self.bottlenecks_gradients[bottleneck_name], {
        self.bottlenecks_tensors[bottleneck_name]: acts,
        self.y_input: [y],
        self.id_:[example]
    })