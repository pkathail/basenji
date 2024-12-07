#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from __future__ import print_function
from optparse import OptionParser

import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import mixed_precision
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()
print(tf.config.list_physical_devices('GPU'))

from basenji import dataset
from basenji import seqnn
from basenji import trainer


"""
basenji_train.py

Train Basenji model using given parameters and data.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <data1_dir> ...'
  parser = OptionParser(usage)
  parser.add_option('-k', dest='keras_fit',
      default=False, action='store_true',
      help='Train with Keras fit method [Default: %default]')
  parser.add_option('-m', dest='mixed_precision',
      default=False, action='store_true',
      help='Train with mixed precision [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='train_out',
      help='Output directory for test statistics [Default: %default]')
  parser.add_option('--restore', dest='restore',
      help='Restore model and continue training [Default: %default]')
  parser.add_option('--transfer', dest='transfer_weights',
      default=None,
      help='')
  parser.add_option('--trunk', dest='trunk',
      default=False, action='store_true',
      help='Restore only model trunk [Default: %default]')
  parser.add_option('--tfr_train', dest='tfr_train_pattern',
      default=None,
      help='Training TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  parser.add_option('--tfr_eval', dest='tfr_eval_pattern',
      default=None,
      help='Evaluation TFR pattern string appended to data_dir/tfrecords for subsetting [Default: %default]')
  parser.add_option('--stats_dir', dest='stats_dir',
      default=None,
      help='Path to statistics.json file [Default: %default]')
  (options, args) = parser.parse_args()
  
  if len(args) < 2:
    parser.error('Must provide parameters and data directory.')
  else:
    params_file = args[0]
    data_dirs = args[1:]

  if options.keras_fit and len(data_dirs) > 1:
    print('Cannot use keras fit method with multi-genome training.')
    exit(1)

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)
  if params_file != '%s/params.json' % options.out_dir:
    shutil.copy(params_file, '%s/params.json' % options.out_dir)

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # read datasets
  train_data = []
  eval_data = []
  strand_pairs = []

  for data_dir in data_dirs:
    # set strand pairs
    targets_df = pd.read_csv('%s/targets.txt'%data_dir, sep='\t', index_col=0)
    if 'strand_pair' in targets_df.columns:
      strand_pairs.append(np.array(targets_df.strand_pair))

    # load train data
    train_data.append(dataset.SeqDataset(data_dir,
    split_label='train',
    batch_size=params_train['batch_size'],
    shuffle_buffer=params_train.get('shuffle_buffer', 128),
    mode='train',
    tfr_pattern=options.tfr_train_pattern,
    phylop=params_train.get('phylop', False),
    target_slice=params_train.get('target_slice', None),
    phylop_smooth=params_train.get('phylop_smooth', None),
    phylop_mask=params_train.get('phylop_mask', False)))

    # load eval data
    eval_data.append(dataset.SeqDataset(data_dir,
    split_label='valid',
    batch_size=params_train['batch_size'],
    mode='eval',
    tfr_pattern=options.tfr_eval_pattern,
    phylop=params_train.get('phylop', False),
    target_slice=params_train.get('target_slice', None),
    phylop_smooth=params_train.get('phylop_smooth', None),
    phylop_mask=params_train.get('phylop_mask', False)))

  params_model['strand_pair'] = strand_pairs

  if options.mixed_precision:
    mixed_precision.set_global_policy('mixed_float16')

  if params_train.get('num_gpu', 1) == 1:
    ########################################
    # one GPU

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)

    # restore
    if options.transfer_weights is not None:
      params_model["seq_depth"] = 4
      seqnn_model_tmp = seqnn.SeqNN(params_model)
      try:
        seqnn_model_tmp.restore(options.restore, trunk=options.trunk)
      except:  # for grouped convs, the model has a different number of layers. For now hard-coding the params
        params_file_tmp = "/global/scratch/users/poojakathail/basenji2/models/params_human.json"
        with open(params_file_tmp) as params_open:
          params = json.load(params_open)
        params_model_tmp = params['model']
        seqnn_model_tmp = seqnn.SeqNN(params_model_tmp)
        seqnn_model_tmp.restore(options.restore, trunk=options.trunk)
      restored_weights = seqnn_model_tmp.model.get_weights()
      untrained_weights = seqnn_model.model.get_weights()
      if options.transfer_weights == "all":
        # if models have same number of layers, concatenate new channel to original model
        if len(restored_weights) == len(untrained_weights):
          print("same number of layers")
          restored_weights[0] = np.concatenate([restored_weights[0], untrained_weights[0][:,4:,:]], axis=1)
          seqnn_model.model.set_weights(restored_weights)
          if params_train.get("freeze_steps", None) is not None:
            print("freezing weights")
            for i, l in enumerate(seqnn_model.model.layers):
              if i not in [4,5]:
                l.trainable = False
        # for grouped convs, model has a different number of layers and more filters. 
        # initialize all layers/nodes possible manually
        else:
          print("different number of layers")
          untrained_weights[0] = restored_weights[0]
          for layer_i in range(2, 6):
            untrained_weights[layer_i][:len(restored_weights[layer_i-1])] = restored_weights[layer_i-1]
          untrained_weights[6][:,:restored_weights[5].shape[1],:] = restored_weights[5]
          for layer_i in range(7, len(untrained_weights)):
            untrained_weights[layer_i] = restored_weights[layer_i-1]
          seqnn_model.model.set_weights(untrained_weights)
      else:  # options.transfer_weights == "first_layer"
        if len(restored_weights) == len(untrained_weights):
          untrained_weights[0] = np.concatenate([restored_weights[0], untrained_weights[0][:,4:,:]], axis=1)
        else:
          untrained_weights[0] = restored_weights[0]
        seqnn_model.model.set_weights(untrained_weights)
    elif options.restore:
      seqnn_model.restore(options.restore, trunk=options.trunk)
    
        
    # initialize trainer
    seqnn_trainer = trainer.Trainer(params_train, train_data, 
                                    eval_data, options.out_dir)

    # compile model
    seqnn_trainer.compile(seqnn_model)

  else:
    ########################################
    # two GPU

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():

      if not options.keras_fit:
        # distribute data
        for di in range(len(data_dirs)):
          train_data[di].distribute(strategy)
          eval_data[di].distribute(strategy)

      # initialize model
      seqnn_model = seqnn.SeqNN(params_model)
        
      # restore
      if options.transfer_weights is not None:
        params_model["seq_depth"] = 4
        seqnn_model_tmp = seqnn.SeqNN(params_model)
        try:
          seqnn_model_tmp.restore(options.restore, trunk=options.trunk)
        except:  # for grouped convs, the model has a different number of layers. For now hard-coding the params
          params_file_tmp = "/global/scratch/users/poojakathail/basenji2/models/params_human.json"
          with open(params_file_tmp) as params_open:
            params = json.load(params_open)
          params_model_tmp = params['model']
          seqnn_model_tmp = seqnn.SeqNN(params_model_tmp)
          seqnn_model_tmp.restore(options.restore, trunk=options.trunk)
          
        restored_weights = seqnn_model_tmp.model.get_weights()
        untrained_weights = seqnn_model.model.get_weights()
        if options.transfer_weights == "all":
          # if models have same number of layers, concatenate new channel to original model
          if len(restored_weights) == len(untrained_weights):
            print("same number of layers")
            restored_weights[0] = np.concatenate([restored_weights[0], untrained_weights[0][:,4:,:]], axis=1)
            seqnn_model.model.set_weights(restored_weights)
            if params_train.get("freeze_steps", None) is not None:
              print("freezing weights")
              for i, l in enumerate(seqnn_model.model.layers):
                if i not in [4,5]:
                  l.trainable = False
          # for grouped convs, model has a different number of layers and more filters. 
          # initialize all layers/nodes possible manually
          else:
            print("different number of layers")
            untrained_weights[0] = restored_weights[0]
            for layer_i in range(2, 6):
              untrained_weights[layer_i][:len(restored_weights[layer_i-1])] = restored_weights[layer_i-1]
            untrained_weights[6][:,:restored_weights[5].shape[1],:] = restored_weights[5]
            for layer_i in range(7, len(untrained_weights)):
              untrained_weights[layer_i] = restored_weights[layer_i-1]
            seqnn_model.model.set_weights(untrained_weights)
        else:  # options.transfer_weights == "first_layer"
          if len(restored_weights) == len(untrained_weights):
            untrained_weights[0] = np.concatenate([restored_weights[0], untrained_weights[0][:,4:,:]], axis=1)
          else:
            untrained_weights[0] = restored_weights[0]
          seqnn_model.model.set_weights(untrained_weights)
      elif options.restore:
        seqnn_model.restore(options.restore, trunk=options.trunk)
      
      # initialize trainer
      seqnn_trainer = trainer.Trainer(params_train, train_data, eval_data, options.out_dir,
                                      strategy, params_train['num_gpu'], options.keras_fit)

      # compile model
      seqnn_trainer.compile(seqnn_model)

  # train model
  if options.keras_fit:
    # if params_train.get("freeze_steps", None) is not None:
    #   untrained_phylop_weights = seqnn_model.model.layers[4].weights[0][:,4,:]
    #   seqnn_trainer.fit_keras(seqnn_model, steps=params_train.get("freeze_steps", None))
    #   trained_phylop_weights = seqnn_model.model.layers[4].weights[0][:,4,:]
    #   if np.all(untrained_phylop_weights == trained_phylop_weights):
    #     print("weights did not get updated", flush=True)

    #   print("unfreezing weights", flush=True)
    #   for i, l in enumerate(seqnn_model.model.layers):
    #     l.trainable = True
    #     if i == 4 and hasattr(l, "frozen_mask"):
    #       # make all channels trainable
    #       l.frozen_mask = tf.constant(np.zeros(l.frozen_mask.shape),dtype=tf.float32)

    #   print("recompiling model", flush=True)
    #   if params_train.get('num_gpu', 1) == 1:
    #     seqnn_trainer.compile(seqnn_model)
    #   else:
    #     with strategy.scope():  
    #       seqnn_trainer.compile(seqnn_model)

    seqnn_trainer.fit_keras(seqnn_model)
  else:
    if len(data_dirs) == 1:
      if params_train.get("freeze_steps", None) is not None:
        untrained_phylop_weights = seqnn_model.model.layers[4].weights[0][:,4,:]
        seqnn_trainer.fit_tape(seqnn_model, steps=params_train.get("freeze_steps", None))
        trained_phylop_weights = seqnn_model.model.layers[4].weights[0][:,4,:]
        if np.all(untrained_phylop_weights == trained_phylop_weights):
          print("weights did not get updated", flush=True)
        
        print("unfreezing weights", flush=True)
        for i, l in enumerate(seqnn_model.model.layers):
          l.trainable = True
          if i == 4 and hasattr(l, "frozen_mask"):
            # make all channels trainable
            l.frozen_mask = tf.constant(np.zeros(l.frozen_mask.shape),dtype=tf.float32)

        print("recompiling model", flush=True)
        if params_train.get('num_gpu', 1) == 1:
          seqnn_trainer.compile(seqnn_model)
        else:
          with strategy.scope():  
            seqnn_trainer.compile(seqnn_model)
      
      seqnn_trainer.fit_tape(seqnn_model)
    else:
      seqnn_trainer.fit2(seqnn_model)

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
