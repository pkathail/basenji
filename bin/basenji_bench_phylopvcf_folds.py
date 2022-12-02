#!/usr/bin/env python
# Copyright 2019 Calico LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
from optparse import OptionParser, OptionGroup
import glob
import h5py
import json
import pdb
import os
import shutil
import sys

import numpy as np
import pandas as pd

import slurm
import util

from basenji_test_folds import stat_tests

"""
basenji_bench_phylopvcf_folds.py

Benchmark Basenji model replicates on PhyloP VCF task.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <exp_dir> <params_file> <data_dir> <phylop_vcf_file>'
  parser = OptionParser(usage)

  # sad options
  sad_options = OptionGroup(parser, 'basenji_sad.py options')
  sad_options.add_option('-f', dest='genome_fasta',
      default='%s/data/hg38.fa' % os.environ['BASENJIDIR'],
      help='Genome FASTA for sequences [Default: %default]')
  sad_options.add_option('-n', dest='norm_file',
      default=None,
      help='Normalize SAD scores')
  sad_options.add_option('-o', dest='out_dir',
      default='sad',
      help='Output directory for tables and plots [Default: %default]')
  sad_options.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  sad_options.add_option('--pseudo', dest='log_pseudo',
      default=1, type='float',
      help='Log2 pseudocount [Default: %default]')
  sad_options.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
  sad_options.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
  sad_options.add_option('--stats', dest='sad_stats',
      default='SAD',
      help='Comma-separated list of stats to save. [Default: %default]')
  sad_options.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  sad_options.add_option('--ti', dest='track_indexes',
      default=None, type='str',
      help='Comma-separated list of target indexes to output BigWig tracks')
  sad_options.add_option('--threads', dest='threads',
      default=False, action='store_true',
      help='Run CPU math and output in a separate thread [Default: %default]')
  sad_options.add_option('-u', dest='penultimate',
      default=False, action='store_true',
      help='Compute SED in the penultimate layer [Default: %default]')

  # regression options
  phylop_options = OptionGroup(parser, 'basenji_bench_phylopvcf.py options')
  phylop_options.add_option('-g', dest='genome',
    default='ce11', help='PhyloP and FASTA genome [Default: %default]')
  phylop_options.add_option('-e', dest='num_estimators',
    default=300, type='int',
    help='Number of random forest estimators [Default: %default]')
  phylop_options.add_option('--msl', dest='msl',
      default=4, type='int',
      help='Random forest min_samples_leaf [Default: %default]')
  parser.add_option_group(phylop_options)

  fold_options = OptionGroup(parser, 'cross-fold options')
  fold_options.add_option('-a', '--alt', dest='alternative',
      default='two-sided', help='Statistical test alternative [Default: %default]')
  fold_options.add_option('-c', dest='crosses',
      default=1, type='int',
      help='Number of cross-fold rounds [Default:%default]')
  fold_options.add_option('--env', dest='conda_env',
      default='tf2.6',
      help='Anaconda environment [Default: %default]')
  fold_options.add_option('--label_exp', dest='label_exp',
      default='Experiment', help='Experiment label [Default: %default]')
  fold_options.add_option('--label_ref', dest='label_ref',
      default='Reference', help='Reference label [Default: %default]')
  fold_options.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  fold_options.add_option('--name', dest='name',
      default='sad', help='SLURM name prefix [Default: %default]')
  fold_options.add_option('-q', dest='queue',
      default='geforce',
      help='SLURM queue on which to run the jobs [Default: %default]')
  fold_options.add_option('-r', dest='ref_dir',
      default=None, help='Reference directory for statistical tests')
  parser.add_option_group(fold_options)

  (options, args) = parser.parse_args()

  if len(args) != 4:
    parser.error('Must provide parameters file and data directory')
  else:
    exp_dir = args[0]
    params_file = args[1]
    data_dir = args[2]
    phylop_vcf_file = args[3]

   # read data parameters
  data_stats_file = '%s/statistics.json' % data_dir
  with open(data_stats_file) as data_stats_open:
    data_stats = json.load(data_stats_open)

  # count folds
  num_folds = len([dkey for dkey in data_stats if dkey.startswith('fold')])

  # genome
  genome_path = os.environ[options.genome.upper()]
  options.genome_fasta = '%s/assembly/%s.fa' % (genome_path, options.genome)

  ################################################################
  # mutation scoring
  ################################################################
  jobs = []
  scores_files = []

  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%d_c%d' % (exp_dir, fi, ci)
      name = '%s-f%dc%d' % (options.name, fi, ci)

      # update output directory
      sad_dir = '%s/%s' % (it_dir, options.out_dir)

      # check if done
      scores_file = '%s/sad.h5' % sad_dir
      scores_files.append(scores_file)
      if os.path.isfile(scores_file):
        print('%s already generated.' % scores_file)
      else:
        basenji_cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
        basenji_cmd += ' conda activate %s;' % options.conda_env
        basenji_cmd += ' echo $HOSTNAME;'

        if options.processes is None:
          basenji_cmd += ' basenji_sad.py'
        else:
          basenji_cmd += ' basenji_sad_multi.py'
          basenji_cmd += ' --max_proc %d' % (options.max_proc // num_folds)
          basenji_cmd += ' -q %s' % options.queue
          basenji_cmd += ' -n %s' % name            
        
        basenji_cmd += ' %s' % options_string(options, sad_options, sad_dir)
        basenji_cmd += ' %s' % params_file
        basenji_cmd += ' %s/train/model_best.h5' % it_dir
        basenji_cmd += ' %s' % phylop_vcf_file
        
        if options.processes is not None:
          jobs.append(basenji_cmd)
        else:
          basenji_job = slurm.Job(basenji_cmd, name,
            out_file='%s.out'%sad_dir,
            err_file='%s.err'%sad_dir,
            cpu=2, gpu=1,
            queue=options.queue,
            mem=30000, time='14-0:00:00')
          jobs.append(basenji_job)
        
  if options.processes is None:
    slurm.multi_run(jobs, verbose=True)
  else:
    util.exec_par(jobs, verbose=True)    

  ################################################################
  # ensemble
  ################################################################
  ensemble_dir = '%s/ensemble' % exp_dir
  if not os.path.isdir(ensemble_dir):
    os.mkdir(ensemble_dir)

  sad_dir = '%s/%s' % (ensemble_dir, options.out_dir)
  if not os.path.isdir(sad_dir):
    os.mkdir(sad_dir)
    
  if not os.path.isfile('%s/sad.h5' % sad_dir):
    print('Generating ensemble scores.')
    ensemble_scores_h5(sad_dir, scores_files)
  else:
    print('Ensemble scores already generated.')

  ################################################################
  # PhyloP regressors
  ################################################################

  jobs = []
  for ci in range(options.crosses):
    for fi in range(num_folds):
      it_dir = '%s/f%d_c%d' % (exp_dir, fi, ci)
      sad_dir = '%s/%s' % (it_dir, options.out_dir)

      if not os.path.isfile('%s/stats.txt' % sad_dir):
        phylop_cmd = 'basenji_bench_phylopvcf.py'
        phylop_cmd += ' -e %d --msl %d' % (options.num_estimators, options.msl)
        phylop_cmd += ' -i 2 -p 4'
        phylop_cmd += ' -o %s' % sad_dir
        phylop_cmd += ' %s/sad.h5' % sad_dir
        phylop_cmd += ' %s' % phylop_vcf_file

        name = '%s-f%dc%d' % (options.name, fi, ci)
        std_pre = '%s/phylop'%sad_dir
        j = slurm.Job(phylop_cmd, name,
                      '%s.out'%std_pre, '%s.err'%std_pre,
                      queue='standard', cpu=4,
                      mem=90000, time='1-0:0:0')
        jobs.append(j)

  # ensemble
  sad_dir = '%s/%s' % (ensemble_dir, options.out_dir)
  if not os.path.isfile('%s/stats.txt' % sad_dir):
    phylop_cmd = 'basenji_bench_phylopvcf.py'
    phylop_cmd += ' -e %d --msl %d' % (options.num_estimators, options.msl)
    phylop_cmd += ' -i 2 -p 4'
    phylop_cmd += ' -o %s' % sad_dir
    phylop_cmd += ' %s/sad.h5' % sad_dir
    phylop_cmd += ' %s' % phylop_vcf_file

    name = '%s-ens' % options.name
    std_pre = '%s/phylop'%sad_dir
    j = slurm.Job(phylop_cmd, name,
                  '%s.out'%std_pre, '%s.err'%std_pre,
                  queue='standard', cpu=4,
                  mem=90000, time='1-0:0:0')
    jobs.append(j)

  slurm.multi_run(jobs, verbose=True)


  ################################################################
  # compare
  ################################################################

  ref_sad_dirs = []
  exp_sad_dirs = []
  for ci in range(options.crosses):
    for fi in range(num_folds):
      exp_sad_dir = '%s/f%d_c%d/%s' % (exp_dir, fi, ci, options.out_dir)
      exp_sad_dirs.append(exp_sad_dir)
      if options.ref_dir is not None:
        ref_sad_dir = '%s/f%d_c%d/%s' % (options.ref_dir, fi, ci, options.out_dir)
        ref_sad_dirs.append(ref_sad_dir)

  exp_pcor_folds, exp_r2_folds = read_metrics(exp_sad_dirs)
  exp_sad_dirs = ['%s/ensemble/%s' % (exp_dir, options.out_dir)]
  exp_pcor_ens, exp_r2_ens = read_metrics(exp_sad_dirs)
  if options.ref_dir is not None:
    ref_pcor_folds, ref_r2_folds = read_metrics(ref_sad_dirs)
    ref_sad_dirs = ['%s/ensemble/%s' % (options.ref_dir, options.out_dir)]
    ref_pcor_ens, ref_r2_ens = read_metrics(ref_sad_dirs)

  print('PearsonR')
  exp_mean = exp_pcor_folds.mean()
  exp_stdm = exp_pcor_folds.std() / np.sqrt(len(exp_pcor_folds))
  expe_mean = exp_pcor_ens.mean()
  expe_stdm = exp_pcor_ens.std() / np.sqrt(len(exp_pcor_ens))
  print('%12s:       %.4f (%.4f)' % (options.label_exp, exp_mean, exp_stdm))
  print('%12s (ens): %.4f (%.4f)' % (options.label_exp, expe_mean, expe_stdm))
  if options.ref_dir is not None:
    ref_mean = ref_pcor_folds.mean()
    ref_stdm = ref_pcor_folds.std() / np.sqrt(len(ref_pcor_folds))
    refe_mean = ref_pcor_ens.mean()
    refe_stdm = ref_pcor_ens.std() / np.sqrt(len(ref_pcor_ens))
    print('%12s:       %.4f (%.4f)' % (options.label_ref, ref_mean, ref_stdm))
    print('%12s (ens): %.4f (%.4f)' % (options.label_ref, refe_mean, refe_stdm))

    mwp, tp = stat_tests(exp_pcor_folds, ref_pcor_folds, options.alternative)
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)
  

  print('\nR2')
  exp_mean = exp_r2_folds.mean()
  exp_stdm = exp_r2_folds.std() / np.sqrt(len(exp_r2_folds))
  expe_mean = exp_r2_ens.mean()
  expe_stdm = exp_r2_ens.std() / np.sqrt(len(exp_r2_ens))
  print('%12s:       %.4f (%.4f)' % (options.label_exp, exp_mean, exp_stdm))
  print('%12s (ens): %.4f (%.4f)' % (options.label_exp, expe_mean, expe_stdm))
  if options.ref_dir is not None:
    ref_mean = ref_r2_folds.mean()
    ref_stdm = ref_r2_folds.std() / np.sqrt(len(ref_r2_folds))
    refe_mean = ref_r2_ens.mean()
    refe_stdm = ref_r2_ens.std() / np.sqrt(len(ref_r2_ens))
    print('%12s:       %.4f (%.4f)' % (options.label_ref, ref_mean, ref_stdm))
    print('%12s (ens): %.4f (%.4f)' % (options.label_ref, refe_mean, refe_stdm))

    mwp, tp = stat_tests(exp_r2_folds, ref_r2_folds, options.alternative)
    print('Mann-Whitney U p-value: %.3g' % mwp)
    print('T-test p-value: %.3g' % tp)


def ensemble_scores_h5(ensemble_dir, scores_files):
  # open ensemble
  ensemble_h5_file = '%s/sad.h5' % ensemble_dir
  if os.path.isfile(ensemble_h5_file):
    os.remove(ensemble_h5_file)
  ensemble_h5 = h5py.File(ensemble_h5_file, 'w')

  # transfer base
  base_keys = ['chr','pos','ref_allele','alt_allele','snp','target_ids','target_labels']
  # skipping ['percentiles', 'SAD_pct']
  scores0_h5 = h5py.File(scores_files[0], 'r')
  for key in scores0_h5.keys():
    if key in base_keys:
      ensemble_h5.create_dataset(key, data=scores0_h5[key])
    elif key == 'SAD':
      sad_shape = scores0_h5[key].shape
  scores0_h5.close()

  # average sum stats
  num_folds = len(scores_files)

  # initialize ensemble array
  sad_values = np.zeros(shape=sad_shape, dtype='float32')

  # read and add folds
  for scores_file in scores_files:
    with h5py.File(scores_file, 'r') as scores_h5:
      sad_values += scores_h5['SAD'][:].astype('float32')
  
  # normalize and downcast
  sad_values /= num_folds
  sad_values = sad_values.astype('float16')

  # save
  ensemble_h5.create_dataset('SAD', data=sad_values)

  ensemble_h5.close()


def options_string(options, group_options, rep_dir):
  options_str = ''

  for opt in group_options.option_list:
    opt_str = opt.get_opt_string()
    opt_value = options.__dict__[opt.dest]

    # wrap askeriks in ""
    if type(opt_value) == str and opt_value.find('*') != -1:
      opt_value = '"%s"' % opt_value

    # no value for bools
    elif type(opt_value) == bool:
      if not opt_value:
        opt_str = ''
      opt_value = ''

    # skip Nones
    elif opt_value is None:
      opt_str = ''
      opt_value = ''

    # modify
    elif opt.dest == 'out_dir':
      opt_value = rep_dir

    options_str += ' %s %s' % (opt_str, opt_value)

  return options_str

def read_metrics(sad_dirs):
  pcor_folds = []
  r2_folds = []

  for sad_dir in sad_dirs:
    pcor_i = np.load('%s/pcor.npy' % sad_dir)
    r2_i = np.load('%s/r2.npy' % sad_dir)

    pcor_folds.append(pcor_i)
    r2_folds.append(r2_i)

  pcor_folds = np.concatenate(pcor_folds)
  r2_folds = np.concatenate(r2_folds)

  return pcor_folds, r2_folds


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
