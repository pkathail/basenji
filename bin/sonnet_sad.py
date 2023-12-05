#!/usr/bin/env python
# Copyright 2020 Calico LLC
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
import pdb
import pickle
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import pysam
import tensorflow as tf
if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

from basenji import dna_io
from basenji import seqnn
from basenji import vcf as bvcf
from basenji_sad import write_snp

'''
sonnet_sad.py

Compute SNP Activity Difference (SAD) scores for SNPs in a VCF file,
using a saved Sonnet model.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <model> <vcf_file>'
  parser = OptionParser(usage)
  parser.add_option('-b', dest='batch_size',
      default=4, type='int',
      help='Batch size [Default: %default]') 
  parser.add_option('-f', dest='genome_fasta',
      default='%s/data/hg19.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SAD scores')
  parser.add_option('-o',dest='out_dir',
      default='sad',
      help='Output directory for tables and plots [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
  parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  parser.add_option('--species', dest='species',
      default='human')
  parser.add_option('--stats', dest='sad_stats',
      default='SAD',
      help='Comma-separated list of stats to save. [Default: %default]')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  parser.add_option('--ti', dest='track_indexes',
      default=None, type='str',
      help='Comma-separated list of target indexes to output BigWig tracks')
  (options, args) = parser.parse_args()

  if len(args) == 2:
    # single worker
    model_file = args[0]
    vcf_file = args[1]

  elif len(args) == 3:
    # multi separate
    options_pkl_file = args[0]
    model_file = args[1]
    vcf_file = args[2]

    # save out dir
    out_dir = options.out_dir

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = out_dir

  elif len(args) == 4:
    # multi worker
    options_pkl_file = args[0]
    model_file = args[1]
    vcf_file = args[2]
    worker_index = int(args[3])

    # load options
    options_pkl = open(options_pkl_file, 'rb')
    options = pickle.load(options_pkl)
    options_pkl.close()

    # update output directory
    options.out_dir = '%s/job%d' % (options.out_dir, worker_index)

  else:
    parser.error('Must provide model and VCF file')

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  if options.track_indexes is None:
    options.track_indexes = []
  else:
    options.track_indexes = [int(ti) for ti in options.track_indexes.split(',')]
    if not os.path.isdir('%s/tracks' % options.out_dir):
      os.mkdir('%s/tracks' % options.out_dir)

  options.shifts = [int(shift) for shift in options.shifts.split(',')]
  options.sad_stats = options.sad_stats.split(',')


  #################################################################
  # read parameters and targets

  if options.targets_file is None:
    target_slice = None
  else:
    targets_df = pd.read_csv(options.targets_file, sep='\t', index_col=0)
    target_ids = targets_df.identifier
    target_labels = targets_df.description
    target_slice = targets_df.index

  #################################################################
  # setup model

  seqnn_model = tf.saved_model.load(model_file).model

  if options.targets_file is None:
    num_targets = 5313
    target_ids = ['t%d' % ti for ti in range(num_targets)]
    target_labels = ['']*len(target_ids)

  #################################################################
  # load SNPs

  # filter for worker SNPs
  if options.processes is not None:
    # determine boundaries
    num_snps = bvcf.vcf_count(vcf_file)
    worker_bounds = np.linspace(0, num_snps, options.processes+1, dtype='int')

    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file, start_i=worker_bounds[worker_index], end_i=worker_bounds[worker_index+1])

  else:
    # read SNPs form VCF
    snps = bvcf.vcf_snps(vcf_file)

  num_snps = len(snps)

  # open genome FASTA
  genome_open = pysam.Fastafile(options.genome_fasta)

  seq_length = seqnn_model.predict_on_batch.input_signature[0].shape[1]
  def snp_gen():
    for snp in snps:
      # get SNP sequences
      snp_1hot_list = bvcf.snp_seq1(snp, seq_length, genome_open)
      for snp_1hot in snp_1hot_list:
        yield snp_1hot

  #################################################################
  # setup output

  sad_out = initialize_output_h5(options.out_dir, options.sad_stats,
                                 snps, target_ids, target_labels)

  #################################################################
  # predict SNP scores, write output

  # initialize predictions stream
  preds_stream = PredStreamGen(seqnn_model, snp_gen(),
  	rc=options.rc, shifts=options.shifts, species=options.species, batch_size=options.batch_size)

  # predictions index
  pi = 0

  for si in range(num_snps):
    # get predictions
    ref_preds = preds_stream[pi]
    pi += 1
    alt_preds = preds_stream[pi]
    pi += 1

    # process SNP
    write_snp(ref_preds, alt_preds, sad_out, si,
              options.sad_stats, options.log_pseudo)

  # close genome
  genome_open.close()

  ###################################################
  # compute SAD distributions across variants

  write_pct(sad_out, options.sad_stats)
  sad_out.close()


def initialize_output_h5(out_dir, sad_stats, snps, target_ids, target_labels):
  """Initialize an output HDF5 file for SAD stats."""

  num_targets = len(target_ids)
  num_snps = len(snps)

  sad_out = h5py.File('%s/sad.h5' % out_dir, 'w')

  # write SNPs
  snp_ids = np.array([snp.rsid for snp in snps], 'S')
  sad_out.create_dataset('snp', data=snp_ids)

  # write SNP chr
  snp_chr = np.array([snp.chr for snp in snps], 'S')
  sad_out.create_dataset('chr', data=snp_chr)

  # write SNP pos
  snp_pos = np.array([snp.pos for snp in snps], dtype='uint32')
  sad_out.create_dataset('pos', data=snp_pos)

  # check flips
  snp_flips = [snp.flipped for snp in snps]

  # write SNP reference allele
  snp_refs = []
  snp_alts = []
  for snp in snps:
    if snp.flipped:
      snp_refs.append(snp.alt_alleles[0])
      snp_alts.append(snp.ref_allele)
    else:
      snp_refs.append(snp.ref_allele)
      snp_alts.append(snp.alt_alleles[0])
  snp_refs = np.array(snp_refs, 'S')
  snp_alts = np.array(snp_alts, 'S')
  sad_out.create_dataset('ref', data=snp_refs)
  sad_out.create_dataset('alt', data=snp_alts)

  # write targets
  sad_out.create_dataset('target_ids', data=np.array(target_ids, 'S'))
  sad_out.create_dataset('target_labels', data=np.array(target_labels, 'S'))

  # initialize SAD stats
  for sad_stat in sad_stats:
    sad_out.create_dataset(sad_stat,
        shape=(num_snps, num_targets),
        dtype='float16',
        compression=None)

  return sad_out


def write_pct(sad_out, sad_stats):
  """Compute percentile values for each target and write to HDF5."""

  # define percentiles
  d_fine = 0.001
  d_coarse = 0.01
  percentiles_neg = np.arange(d_fine, 0.1, d_fine)
  percentiles_base = np.arange(0.1, 0.9, d_coarse)
  percentiles_pos = np.arange(0.9, 1, d_fine)

  percentiles = np.concatenate([percentiles_neg, percentiles_base, percentiles_pos])
  sad_out.create_dataset('percentiles', data=percentiles)
  pct_len = len(percentiles)

  for sad_stat in sad_stats:
    sad_stat_pct = '%s_pct' % sad_stat

    # compute
    sad_pct = np.percentile(sad_out[sad_stat], 100*percentiles, axis=0).T
    sad_pct = sad_pct.astype('float16')

    # save
    sad_out.create_dataset(sad_stat_pct, data=sad_pct, dtype='float16')


class PredStreamGen:
  """ Interface to acquire predictions via a buffered stream mechanism
        rather than getting them all at once and using excessive memory.
        Accepts generator and constructs stream batches from it. """
  def __init__(self, model, seqs_gen, batch_size=4, stream_size=32,
               rc=False, shifts=[0], species='human', verbose=False):
    self.model = model
    self.seqs_gen = seqs_gen
    self.batch_size = batch_size
    self.stream_size = stream_size
    self.rc = rc
    self.shifts = shifts
    self.ensembled = len(self.shifts) + int(self.rc)*len(self.shifts)
    self.species = species
    self.verbose = verbose

    self.stream_start = 0
    self.stream_end = 0


  def __getitem__(self, i):
    # acquire predictions, if needed
    if i >= self.stream_end:
      # update start
      self.stream_start = self.stream_end

      if self.verbose:
        print('Predicting from %d' % self.stream_start, flush=True)

      # get next sequences
      seqs_1hot = self.next_seqs()

      # predict stream
      stream_preds = []
      si = 0
      while si < seqs_1hot.shape[0]:
        spreds = self.model.predict_on_batch(seqs_1hot[si:si+self.batch_size])
        spreds = spreds[self.species].numpy()
        stream_preds.append(spreds)
        si += self.batch_size
      stream_preds = np.concatenate(stream_preds, axis=0)

      # average ensemble
      ens_seqs, seq_len, num_targets = stream_preds.shape
      num_seqs = ens_seqs // self.ensembled
      stream_preds = np.reshape(stream_preds,
          (num_seqs, self.ensembled, seq_len, num_targets))
      self.stream_preds = stream_preds.mean(axis=1)

      # update end
      self.stream_end = self.stream_start + self.stream_preds.shape[0]

    return self.stream_preds[i - self.stream_start]

  def next_seqs(self):
    """ Construct array of sequences for this stream chunk. """

    # extract next sequences from generator
    seqs_1hot = []
    stream_end = self.stream_start+self.stream_size
    for si in range(self.stream_start, stream_end):
      try:
        seqs_1hot.append(self.seqs_gen.__next__())
      except StopIteration:
        continue

    # initialize ensemble
    seqs_1hot_ens = []

    # add rc/shifts
    for seq_1hot in seqs_1hot:
      for shift in self.shifts:
        seq_1hot_aug = dna_io.hot1_augment(seq_1hot)
        seqs_1hot_ens.append(seq_1hot_aug)
        if self.rc:
          seq_1hot_aug = dna_io.hot1_rc(seq_1hot_aug)
          seqs_1hot_ens.append(seq_1hot_aug)

    seqs_1hot_ens = np.array(seqs_1hot_ens, dtype='float32')
    return seqs_1hot_ens

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
