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
import os
from glob import glob
from natsort import natsorted

import h5py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

'''
basenji_predict_bed.py

Predict sequences from a BED file.
'''

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options]'
  parser = OptionParser(usage)
  parser.add_option('-c', dest='test_chrs',
      default=None, help='Comma-separated list of test chromosomes')
  parser.add_option('-p', dest='predict_dir',
      default=None, help='Directory containing predictions of bed regions')
  parser.add_option('-d', dest='data_dir',
      default=None, help='Directory containing processed training and test data')
  parser.add_option('-o', dest='out_dir',
      default='pred_out',
      help='Output directory [Default: %default]')
  parser.add_option('-r', dest='rescale',
      default=False, action='store_true',
      help='Rescale targets and predictions')
  parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
  (options, args) = parser.parse_args()

  TARGET_SIGNAL_DIR = "/clusterfs/nilah/pooja/kidney_data/CellTypeSpecificPeakClusters_ArchR_clust11/model_predictions"

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)

  #################################################################
  # collect target information

  if options.targets_file is None:
    target_slice = None
    cell_types = None
  else:
    targets_df = pd.read_table(options.targets_file, index_col=0)
    target_slice = targets_df.index
    cell_types = targets_df["identifier"].values

  #################################################################

  if options.rescale:
    seqs = pd.read_csv(f"{options.data_dir}/sequences.bed", sep="\t", names=["chr", "start", "stop", "fold"])
    num_train_seqs = seqs[seqs["fold"] == "train"].shape[0]
    seqs_cov = []
    for i in target_slice[:len(target_slice)-1]:
        seqs_cov.append(h5py.File(f"{options.data_dir}/seqs_cov/{i}.h5", "r")["targets"][:num_train_seqs, :])
    seqs_cov = np.hstack(seqs_cov)
    rescale_factor = np.max(np.abs(seqs_cov))

  all_chrs = list(range(1, 23))
  if options.test_chrs is not None:
    options.test_chrs = options.test_chrs.split(',')
  else:
    options.test_chrs = all_chrs
  peak_types = natsorted([p.split("/")[-1] for p in glob(f"{options.predict_dir}/cluster*")])

  peak_bed_preds = {}
  for pt in peak_types:
    for i, chr in enumerate(all_chrs):
      preds = h5py.File(f"{options.predict_dir}/{pt}/chr{chr}/predict.h5", "r")
      preds_df = pd.DataFrame(np.squeeze(preds["preds"][:,:,:]), columns=[f"{ct}_pred" for ct in cell_types])  
      preds_df["chrom"] = preds["chrom"][:].astype(str)
      preds_df["start"] = preds["start"][:]
      preds_df["end"] = preds["end"][:]
    
      if i == 0:
        peak_bed_preds[pt] = preds_df
      else:
        peak_bed_preds[pt] = pd.concat([peak_bed_preds[pt], preds_df])
    peak_bed_preds[pt]["name"] = [f"{pt}_{i}" for i in range(peak_bed_preds[pt].shape[0])]
  
  for pt in peak_types:
    for ct in cell_types:
      if ct != "Mean":
        peak_bed_preds[pt][f"{ct}_target"] = pd.read_csv(f"{TARGET_SIGNAL_DIR}/{pt}/{ct}_target_signal.out", 
                                                         sep="\t",
                                                         names=["name", "size", "covered", "sum", "mean0", "mean"])["sum"].values
    peak_bed_preds[pt][f"Mean_target"] = peak_bed_preds[pt][[f"{ct}_target" for ct in cell_types if ct != "Mean"]].mean(axis=1)

    # subset to test chromosomes
    peak_bed_preds[pt] = peak_bed_preds[pt][peak_bed_preds[pt]["chrom"].isin(options.test_chrs)]
  
  # rescale targets and predictions
  if options.rescale:
    for pt in peak_types:
        for ct in cell_types:
            if ct != "Mean":
                peak_bed_preds[pt][f"{ct}_pred"] = peak_bed_preds[pt][f"{ct}_pred"]*rescale_factor + peak_bed_preds[pt][f"Mean_pred"]
  
  # calculate performance
  log_performance_by_peak_type = pd.DataFrame([], index=peak_types, columns=cell_types)
  for peak_type in peak_types:
    for cell_type in cell_types:
        log_performance_by_peak_type.loc[peak_type, cell_type] = pearsonr(np.log2(peak_bed_preds[peak_type][f"{cell_type}_target"]+1),
                                                                      np.log2(peak_bed_preds[peak_type][f"{cell_type}_pred"]+1))[0]
  # bar plot of all peak types
  cmap = plt.get_cmap('tab10')
  target_order = [2, 4, 8, 9, 7, 3, 6, 1, 5, 0, 10]
  fig, ax = plt.subplots(2, 6, figsize=(20,8))
  for i, peak_type in enumerate(peak_types):
    ax[i//6,i%6].bar(np.arange(len(cell_types)), 
                     log_performance_by_peak_type.loc[peak_type].values[target_order],
                     color=[cmap(ti) for ti in target_order])
    ax[i//6,i%6].set_ylim(0, 0.75)
    ax[i//6,i%6].set_title(peak_type)
    ax[i//6,i%6].set_ylabel("log-log Pearson R")
    ax[i//6,i%6].set_xticks(np.arange(len(cell_types)))
    l = ax[i//6,i%6].set_xticklabels(cell_types[target_order], rotation=45)
  plt.tight_layout()
  plt.savefig(f"{options.out_dir}/pearsonr_by_peak_type.pdf")
  plt.close()

  # bar plot of specific cell types
  sample_size = 5000
  cmap = plt.get_cmap('tab10')
  cell_type_peak_set_mapping = {"CFH": ["cluster6_PanTubule", "cluster3_Ubiquitous"],
                              "PT": ["cluster11_PT", "cluster6_PanTubule", "cluster3_Ubiquitous"],
                              "LOH": ["cluster4_DistalNephron", "cluster6_PanTubule", "cluster3_Ubiquitous"],
                              "DT": ["cluster4_DistalNephron", "cluster6_PanTubule", "cluster3_Ubiquitous"],
                              "CD": ["cluster1_CD", "cluster4_DistalNephron", "cluster6_PanTubule", "cluster3_Ubiquitous"]}

  fig, ax = plt.subplots(1, len(cell_type_peak_set_mapping), figsize=(5*len(cell_type_peak_set_mapping), 5))

  for ti, (ct, peak_sets) in enumerate(cell_type_peak_set_mapping.items()):
    ax[ti].bar(np.arange(len(peak_sets)), 
               [log_performance_by_peak_type.loc[peak_set, ct] for peak_set in peak_sets],
               color=[cmap(i) for i in range(len(peak_sets))])
    ax[ti].set_ylim(0, 0.85)
    ax[ti].set_title(ct)
    ax[ti].set_ylabel("log-log Pearson R")
    ax[ti].set_xticks(np.arange(len(peak_sets)))
    l = ax[ti].set_xticklabels([p.split("_")[1] for p in peak_sets], rotation=45)
  plt.tight_layout()
  plt.savefig(f"{options.out_dir}/pearsonr_by_cell_type.pdf")
  plt.close()

  # scatter plots of specific cell types
  for ti, (ct, peak_sets) in enumerate(cell_type_peak_set_mapping.items()):
    
    fig, ax = plt.subplots(1, len(peak_sets), figsize=(7*len(peak_sets), 7))
    
    for i, peak_set in enumerate(peak_sets):
        sample_inds = np.random.choice(np.arange(peak_bed_preds[peak_set].shape[0]), sample_size, replace=False)
        # subset and flatten
        test_targets_ti_flat = peak_bed_preds[peak_set][f"{ct}_target"].values[sample_inds].flatten(
          ).astype('float32')
        test_preds_ti_flat = peak_bed_preds[peak_set][f"{ct}_pred"].values[sample_inds].flatten().astype(
              'float32')

        # take log2
        test_targets_ti_log = np.log2(test_targets_ti_flat + 1)
        test_preds_ti_log = np.log2(test_preds_ti_flat + 1)

        ax[i] = scatter_plot(test_targets_ti_log, test_preds_ti_log, peak_set, 
                             ax[i], nonzero_pearson=True)

        sns.despine()
    fig.suptitle(ct, fontsize=28)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{options.out_dir}/{ct}_scatter.pdf")
    plt.close()

  # scatter plots of specific cell types with matched peak heights
  bins = np.arange(2.5, 10.5, .5)
  for ti, (ct, peak_sets) in enumerate(cell_type_peak_set_mapping.items()):
    
    fig, ax = plt.subplots(2, len(peak_sets), figsize=(7*len(peak_sets), 14))
    
    for i, peak_set in enumerate(peak_sets):
        # subset and flatten
        test_targets_ti_flat = peak_bed_preds[peak_set][f"{ct}_target"].values.flatten(
          ).astype('float32')
        test_preds_ti_flat = peak_bed_preds[peak_set][f"{ct}_pred"].values.flatten().astype(
              'float32')

        # take log2
        test_targets_ti_log = np.log2(test_targets_ti_flat + 1)
        test_preds_ti_log = np.log2(test_preds_ti_flat + 1)
        
        if i == 0:
            bin_counts, bins = np.histogram(np.log2(peak_bed_preds[peak_set][f"{ct}_target"].values + 1), 
                                            bins=bins)
            bin_counts = ((bin_counts/bin_counts.sum())*sample_size).astype(int)
            
        bin_assignments = np.digitize(test_targets_ti_log, bins) - 1
        sample_inds = []
        for bin_i, bin_count in enumerate(bin_counts):
            bin_inds = np.where(bin_assignments == bin_i)[0]
            try:
                sampled_bin_inds = np.random.choice(bin_inds, bin_count, replace=False)
            except:
                if len(bin_inds) > 0:
                    #print(ct, peak_set, bin_i, 0)
                    sampled_bin_inds = np.random.choice(bin_inds, bin_count, replace=True)
                else:
                    #print(ct, peak_set, bin_i, 1)
                    sampled_bin_inds = np.array([])
            sample_inds.extend(sampled_bin_inds)
        sample_inds = np.array(sample_inds)
        
        unmatched_sample_inds = np.random.choice(np.arange(peak_bed_preds[peak_set].shape[0]), sample_size, replace=False)
        ax[0, i] = scatter_plot(test_targets_ti_log[unmatched_sample_inds], test_preds_ti_log[unmatched_sample_inds], peak_set + " unmatched", 
                             ax[0, i], nonzero_pearson=True, xlim=(0,12), ylim=(0,12), alpha=0.2)
        ax[1, i] = scatter_plot(test_targets_ti_log[sample_inds], test_preds_ti_log[sample_inds], peak_set + " matched", 
                             ax[1, i], nonzero_pearson=True, xlim=(0,12), ylim=(0,12), alpha=0.2)

        sns.despine()
    fig.suptitle(ct, fontsize=28)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{options.out_dir}/{ct}_scatter_matched_peak_height.pdf")
    plt.close()


################################################################################
# Plotting helpers
################################################################################

def scatter_lims(vals1, vals2=None, buffer=.05):
    if vals2 is not None:
        vals = np.concatenate((vals1, vals2))
    else:
        vals = vals1
    vmin = np.nanmin(vals)
    vmax = np.nanmax(vals)

    buf = .05 * (vmax - vmin)

    if vmin == 0:
        vmin -= buf / 2
    else:
        vmin -= buf
    vmax += buf

    return vmin, vmax

def scatter_plot(targets, preds, name, ax, alpha=0.5, nonzero_pearson=False, xlim=None, ylim=None):
    sns.set(font_scale=1.2, style='ticks')
    gold = sns.color_palette('husl', 8)[1]
    sns.regplot(targets,preds, color='black',
                order=1,
                scatter_kws={'s': 10,
                             'alpha': alpha},
                line_kws={'color': gold},
                ax=ax)
    
    if not xlim:
        xmin, xmax = scatter_lims(targets)
    else:
        xmin, xmax = xlim
    if not ylim:
        ymin, ymax = scatter_lims(preds)
    else:
        ymin, ymax = ylim
        
    ax.set_title(name, fontsize=24)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('log2 Experiment')
    ax.set_ylabel('log2 Prediction')
    
    corr, csig = pearsonr(targets, preds)
    corr_str = 'PearsonR: %.3f' % corr
    xlim_eps = (xmax - xmin) * .03
    ylim_eps = (ymax - ymin) * .05

    ax.text(xmin + xlim_eps,
                             ymin + ylim_eps,
                             corr_str,
                             horizontalalignment='left',
                             fontsize=20)
    if nonzero_pearson:
        nonzero_inds = np.where(targets > 0)[0]
        nonzero_corr, nonzero_csig = pearsonr(targets[nonzero_inds], preds[nonzero_inds])
        nonzero_corr_str = 'PearsonR (nonzero): %.3f' % nonzero_corr
        ax.text(xmin + xlim_eps,
                             ymin + 3 * ylim_eps,
                             nonzero_corr_str,
                             horizontalalignment='left',
                             fontsize=20)
    return ax

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
