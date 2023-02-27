from optparse import OptionParser
import numpy as np
import pandas as pd
import h5py
import scipy
from scipy import io
import scanpy as sc
import anndata
import pkg_resources
from basenji import seqnn
from Bio import SeqIO
# from scbasset.utils import *
from scbasset.basenji_utils import *
from scipy.stats import pearsonr

# plotting functions
import seaborn as sns
import matplotlib.pyplot as plt

import os
import subprocess
import shutil
import json
from glob import glob
from joblib import Parallel, delayed
from multiprocessing import cpu_count


def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <motif_fasta_folder>'
    parser = OptionParser(usage)
    parser.add_option('-o',dest='out_dir',
      default=None,
      help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-t', dest='targets_file',
      default=None, type='str',
      help='File specifying target indexes and labels in table format')
    parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Average forward and reverse complement predictions [Default: %default]')
    parser.add_option('--shifts', dest='shifts',
      default='0', type='str',
      help='Ensemble prediction shifts [Default: %default]')
    (options, args) = parser.parse_args()

    if len(args) == 3:
    	params_file = args[0]
    	model_file = args[1]
    	motif_fasta_folder = args[2]
    else:
    	parser.error('Must provide parameters file, model file, and motif fasta folder.')

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    targets = pd.read_csv(options.targets_file, sep="\t")

    options.shifts = [int(shift) for shift in options.shifts.split(',')]

    # read model parameters
    with open(params_file) as params_open:
        params = json.load(params_open)
        params_model = params['model']
        params_train = params['train']

    # initialize model
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    # compute tf activity scores
    tfs = os.listdir(f"{motif_fasta_folder}/shuffled_peaks_motifs/")
    tfs = [tf.split(".fasta")[0] for tf in tfs]
    tf_activity_scores = pd.DataFrame([], index=tfs, columns=targets["identifier"].values)
    for tf in tfs:
        scores = motif_score(tf, seqnn_model, motif_fasta_folder=motif_fasta_folder)
        tf_activity_scores.loc[tf] = scores
    tf_activity_scores.to_csv(f"{options.out_dir}/tf_activity_scores.tsv", sep="\t", header=True, index=True)

def pred_on_fasta(fa, model):
    """Run a trained model on a fasta file.
    Args:
        fa:             fasta file to run on. Need to have a fixed size of 1344. Default
                        sequence size of trained model.
        model:          a trained scBasset model.
    Returns:
        array:          a peak*cell imputed accessibility matrix. Sequencing depth corrected for.
    """
    records = list(SeqIO.parse(fa, "fasta"))
    seqs = [str(i.seq) for i in records]
    seqs_1hot = np.array([dna_1hot(i) for i in seqs])
    pred = model.predict(seqs_1hot)
    return pred


def motif_score(tf, model, motif_fasta_folder):
    """score motifs for any given TF.
    Args:
        tf:             TF of interest. By default we only provide TFs to score in
                        https://storage.googleapis.com/scbasset_tutorial_data/Homo_sapiens_motif_fasta.tar.gz.
                        To score on additional motifs, follow make_fasta.R in the tarball 
                        to create dinucleotide shuffled sequences with and without motifs of
                        interest.
        model:          a trained scBasset model.
        motif_fasta_folder: folder for dinucleotide shuffled sequences with and without any motif.
                        We provided motifs from CIS-BP/Homo_sapiens.meme downloaded from the
                        MEME Suite (https://meme-suite.org/meme/) in 
                        https://storage.googleapis.com/scbasset_tutorial_data/Homo_sapiens_motif_fasta.tar.gz.
    Returns:
        array:          a vector for motif activity per cell. (cell order is the
                        same order as the model.)
    """
    fasta_motif = "%s/shuffled_peaks_motifs/%s.fasta" % (motif_fasta_folder, tf)
    fasta_bg = "%s/shuffled_peaks.fasta" % motif_fasta_folder

    pred_motif = pred_on_fasta(fasta_motif, model)
    pred_bg = pred_on_fasta(fasta_bg, model)
    tf_score = pred_motif.mean(axis=0) - pred_bg.mean(axis=0)
#     tf_score = (tf_score - tf_score.mean()) / tf_score.std()
    return tf_score


if __name__ == '__main__':
    main()

