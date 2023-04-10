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
import gc
import json
import os
from queue import Queue
import sys
from threading import Thread

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import rel_entr
import seaborn as sns
import tensorflow as tf

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import dna_io
from basenji import seqnn
from basenji import stream
from basenji import vcf

from basenji_sat_bed import ScoreWorker, satmut_gen

'''
basenji_sat_vcf.py

Perform an in silico saturated mutagenesis of the sequences surrounding variants
given in a VCF file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-d', dest='mut_down',
      default=0, type='int',
      help='Nucleotides downstream of center sequence to mutate [Default: %default]')
  parser.add_option('-f', dest='figure_width',
      default=20, type='float',
      help='Figure width [Default: %default]')
  parser.add_option('--f1', dest='genome1_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA which which major allele sequences will be drawn')
  parser.add_option('--f2', dest='genome2_fasta',
      default=None,
      help='Genome FASTA which which minor allele sequences will be drawn')
  parser.add_option('-l', dest='mut_len',
      default=200, type='int',
      help='Length of centered sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_vcf',
      help='Output directory [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--stats', dest='sad_stats',
      default='sum',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('-u', dest='mut_up',
      default=0, type='int',
      help='Nucleotides upstream of center sequence to mutate [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    parser.error('Must provide parameters and model files and VCF')
  else:
    params_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.sad_stats = [sad_stat.lower() for sad_stat in options.sad_stats.split(',')]

  if options.mut_up > 0 or options.mut_down > 0:
    options.mut_len = options.mut_up + options.mut_down
  else:
    assert(options.mut_len > 0)
    options.mut_up = options.mut_len // 2
    options.mut_down = options.mut_len - options.mut_up

  #################################################################
  # read parameters and targets

  # read model parameters
  with open(params_file) as params_open:
    params = json.load(params_open)
  params_model = params['model']
  params_train = params['train']

  # read targets
  if options.targets_file is None:
    target_slice = None
  else:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_slice = targets_df.index

  #################################################################
  # setup model

  seqnn_model = seqnn.SeqNN(params_model)
  seqnn_model.restore(model_file)
  seqnn_model.build_slice(target_slice)
  seqnn_model.build_ensemble(options.rc, options.shifts)

  num_targets = seqnn_model.num_targets()

  #################################################################
  # SNP sequence dataset

  # load SNPs
  snps = vcf.vcf_snps(vcf_file)

  # get one hot coded input sequences
  if not options.genome2_fasta:
    seqs_1hot, seq_headers, snps, seqs_dna = vcf.snps_seq1(
        snps, params_model['seq_length'], options.genome1_fasta, return_seqs=True)
  else:
    seqs_1hot, seq_headers, snps, seqs_dna = vcf.snps2_seq1(
        snps, params_model['seq_length'], options.genome1_fasta,
        options.genome2_fasta, return_seqs=True)
  num_seqs = seqs_1hot.shape[0]

  # determine mutation region limits
  seq_mid = params_model['seq_length'] // 2
  mut_start = seq_mid - options.mut_up
  mut_end = mut_start + options.mut_len

  # make sequence generator
  seqs_gen = satmut_gen(seqs_dna, mut_start, mut_end)

  #################################################################
  # setup output

  scores_h5_file = '%s/scores.h5' % options.out_dir
  if os.path.isfile(scores_h5_file):
    os.remove(scores_h5_file)
  scores_h5 = h5py.File(scores_h5_file, 'w')
  scores_h5.create_dataset('label',
    data=np.array(seq_headers, dtype='S'))
  scores_h5.create_dataset('seqs', dtype='bool',
    shape=(num_seqs, options.mut_len, 4))
  for sad_stat in options.sad_stats:
    scores_h5.create_dataset(sad_stat, dtype='float16',
        shape=(num_seqs, options.mut_len, 4, num_targets))

  preds_per_seq = 1 + 3*options.mut_len

  score_threads = []
  score_queue = Queue()
  for i in range(1):
    sw = ScoreWorker(score_queue, scores_h5, options.sad_stats,
                     mut_start, mut_end)
    sw.start()
    score_threads.append(sw)

  #################################################################
  # predict scores and write output

  # find center
  preds_length = seqnn_model.target_lengths[0]
  center_start = preds_length // 2
  if preds_length % 2 == 0:
    center_end = center_start + 2
  else:
    center_end = center_start + 1

  # initialize predictions stream
  preds_stream = stream.PredStreamGen(seqnn_model, seqs_gen, params_train['batch_size'])

  # predictions index
  pi = 0

  for si in range(num_seqs):
    print('Predicting %d' % si, flush=True)

    # collect sequence predictions
    seq_preds_sum = []
    seq_preds_center = []
    seq_preds_scd = []
    seq_preds_js = []
    preds_mut0 = preds_stream[pi]
    preds_mut0_norm = preds_mut0 / preds_mut0.sum(axis=0)
    for spi in range(preds_per_seq):
      preds_mut = preds_stream[pi]
      preds_sum = preds_mut.sum(axis=0)
      seq_preds_sum.append(preds_sum)
      if 'center' in options.sad_stats:
        preds_center = preds_mut[center_start:center_end,:].sum(axis=0)
        seq_preds_center.append(preds_center)
      if 'scd' in options.sad_stats:
        preds_scd = np.sqrt(((preds_mut-preds_mut0)**2).sum(axis=0))
        seq_preds_scd.append(preds_scd)
      if 'js' in options.sad_stats:
        preds_mut_norm = preds_mut / preds_mut.sum(axis=0)
        ref_alt_entr = rel_entr(preds_mut0_norm, preds_mut_norm).sum(axis=0)
        alt_ref_entr = rel_entr(preds_mut_norm, preds_mut0_norm).sum(axis=0)
        js_dist = (ref_alt_entr + alt_ref_entr) / 2
        seq_preds_js.append(js_dist)

      pi += 1
    seq_preds_sum = np.array(seq_preds_sum)
    seq_preds_center = np.array(seq_preds_center)
    seq_preds_scd = np.array(seq_preds_scd)
    seq_preds_js = np.array(seq_preds_js)

    # wait for previous to finish
    score_queue.join()

    # queue sequence for scoring
    seq_pred_stats = (seq_preds_sum, seq_preds_center, seq_preds_scd, seq_preds_js)
    score_queue.put((seqs_dna[si], seq_pred_stats, si))
    
    gc.collect()

  # finish queue
  print('Waiting for threads to finish.', flush=True)
  score_queue.join()

  # close output HDF5
  scores_h5.close()


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
  # pdb.runcall(main)
