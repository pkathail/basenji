#!/usr/bin/env python
# Copyright 2017 Calico LLC

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

from optparse import OptionParser

import glob
import os
import pickle
import shutil
import subprocess
import sys

import h5py
import numpy as np

import slurm

"""
basenji_sat_bed_multi.py

Perform an in silico saturation mutagenesis of sequences in a BED file,
using multiple processes.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <params_file> <model_file> <bed_file>'
  parser = OptionParser(usage)

  # basenji_sat_bed.py options
  parser.add_option('-d', dest='mut_down',
      default=0, type='int',
      help='Nucleotides downstream of center sequence to mutate [Default: %default]')
  parser.add_option('-f', dest='genome_fasta',
      default=None,
      help='Genome FASTA for sequences [Default: %default]')
  parser.add_option('-l', dest='mut_len',
      default=200, type='int',
      help='Length of center sequence to mutate [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='sat_mut', help='Output directory [Default: %default]')
  parser.add_option('--plots', dest='plots',
      default=False, action='store_true',
      help='Make heatmap plots [Default: %default]')
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

  # _multi.py options
  parser.add_option('-e', dest='conda_env',
      default='tf2.6',
      help='Anaconda environment [Default: %default]')
  parser.add_option('--max_proc', dest='max_proc',
      default=None, type='int',
      help='Maximum concurrent processes [Default: %default]')
  parser.add_option('-n', dest='name',
      default='sat',
      help='SLURM job name prefix [Default: %default]')
  parser.add_option('-p', dest='processes',
      default=None, type='int',
      help='Number of processes, passed by multi script')
  parser.add_option('-q', dest='queue',
      default='geforce',
      help='SLURM queue on which to run the jobs [Default: %default]')
  parser.add_option('-r', '--restart', dest='restart',
      default=False, action='store_true',
      help='Restart a partially completed job [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 3:
    print(args)
    parser.error('Must provide parameters and model files and BED file')
  else:
    params_file = args[0]
    model_file = args[1]
    bed_file = args[2]

  #######################################################
  # prep work

  # output directory
  if not options.restart:
    if os.path.isdir(options.out_dir):
      print('Please remove %s' % options.out_dir, file=sys.stderr)
      exit(1)
    os.mkdir(options.out_dir)

  # pickle options
  options_pkl_file = '%s/options.pkl' % options.out_dir
  options_pkl = open(options_pkl_file, 'wb')
  pickle.dump(options, options_pkl)
  options_pkl.close()

  #######################################################
  # launch worker threads
  jobs = []
  for pi in range(options.processes):
    if not options.restart or not job_completed(options, pi):
      cmd = '. /home/drk/anaconda3/etc/profile.d/conda.sh;'
      cmd += ' conda activate %s;' % options.conda_env

      cmd += ' basenji_sat_bed.py %s %s %d' % (
          options_pkl_file, ' '.join(args), pi)
      name = '%s_p%d' % (options.name, pi)
      outf = '%s/job%d.out' % (options.out_dir, pi)
      errf = '%s/job%d.err' % (options.out_dir, pi)
      j = slurm.Job(cmd, name,
          outf, errf,
          queue=options.queue,
          cpu=2, gpu=1,
          mem=30000, time='14-0:0:0')
      jobs.append(j)

  slurm.multi_run(jobs, max_proc=options.max_proc, verbose=True,
                  launch_sleep=10, update_sleep=60)

  #######################################################
  # collect output

  sad_stat = options.sad_stats.split(',')[0]
  collect_h5(options.out_dir, options.processes, sad_stat)

  # for pi in range(options.processes):
  #     shutil.rmtree('%s/job%d' % (options.out_dir,pi))


def collect_h5(out_dir, num_procs, sad_stat):
  h5_file = 'scores.h5'

  # count sequences
  num_seqs = 0
  for pi in range(num_procs):
    # open job
    job_h5_file = '%s/job%d/%s' % (out_dir, pi, h5_file)
    job_h5_open = h5py.File(job_h5_file, 'r')
    num_seqs += job_h5_open[sad_stat].shape[0]
    seq_len = job_h5_open[sad_stat].shape[1]
    num_targets = job_h5_open[sad_stat].shape[-1]
    job_h5_open.close()

  # initialize final h5
  final_h5_file = '%s/%s' % (out_dir, h5_file)
  final_h5_open = h5py.File(final_h5_file, 'w')

  # keep dict for string values
  final_strings = {}

  job0_h5_file = '%s/job0/%s' % (out_dir, h5_file)
  job0_h5_open = h5py.File(job0_h5_file, 'r')
  for key in job0_h5_open.keys():
    key_shape = list(job0_h5_open[key].shape)
    key_shape[0] = num_seqs
    key_shape = tuple(key_shape)
    if job0_h5_open[key].dtype.char == 'S':
      final_strings[key] = []
    else:
      final_h5_open.create_dataset(key, shape=key_shape, dtype=job0_h5_open[key].dtype)

  # set values
  si = 0
  for pi in range(num_procs):
    # open job
    job_h5_file = '%s/job%d/%s' % (out_dir, pi, h5_file)
    job_h5_open = h5py.File(job_h5_file, 'r')

    # append to final
    for key in job_h5_open.keys():
        job_seqs = job_h5_open[key].shape[0]
        if job_h5_open[key].dtype.char == 'S':
          final_strings[key] += list(job_h5_open[key])
        else:
          final_h5_open[key][si:si+job_seqs] = job_h5_open[key]

    job_h5_open.close()
    si += job_seqs

  # create final string datasets
  for key in final_strings:
    final_h5_open.create_dataset(key,
      data=np.array(final_strings[key], dtype='S'))

  final_h5_open.close()


def job_completed(options, pi):
  """Check whether a specific job has generated its
     output file."""
  out_file = '%s/job%d/scores.h5' % (options.out_dir, pi)
  valid_file = True
  if not os.path.isfile(out_file):
    valid_file = False
  else:
    try:
      out_open = h5py.File(out_file, 'r')
    except OSError:
      valid_file = False
  return valid_file


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
