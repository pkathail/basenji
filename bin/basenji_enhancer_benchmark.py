from optparse import OptionParser
import os
import json
import random
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import pandas as pd
import pysam 
import pyBigWig
from sklearn.metrics import average_precision_score

from basenji import seqnn
from basenji import dna_io

def main():
    usage = 'usage: %prog [options] <params_file> <model_file>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='genome_fasta',
      default="/clusterfs/nilah/pooja/genomes/hg38.ml.fa",
      help='Genome FASTA for sequences [Default: %default]')
    parser.add_option('--ph', dest='phylop_bigwig',
      default="/clusterfs/nilah/ruchir/data/conservation/241-mammalian-2020v2.bigWig",
      help='Genome FASTA for sequences [Default: %default]')
    parser.add_option('-o', dest='out_dir',
      default='pred_out',
      help='Output directory [Default: %default]')
    parser.add_option('--rc', dest='rc',
      default=False, action='store_true',
      help='Ensemble forward and reverse complement predictions [Default: %default]')
    parser.add_option('--shifts', dest='shifts',
      default='0',
      help='Ensemble prediction shifts [Default: %default]')
    parser.add_option('--n', dest='n',
      default=1, type='int',
      help='N shuffled seq predictions to make')
    parser.add_option('--phylop_smooth', dest='phylop_smooth',
      default="1", type='str',
      help='')
    (options, args) = parser.parse_args()
    if len(args) == 2:
        params_file = args[0]
        model_file = args[1]
    else:
        parser.error("Must provide parameter and model files")

    os.makedirs(options.out_dir, exist_ok=True)
    options.shifts = [int(shift) for shift in options.shifts.split(',')]
    options.phylop_smooth = [int(p) for p in options.phylop_smooth.split(",")]
    
    with open(params_file) as params_open:
        params = json.load(params_open)
    params_model = params['model']
    params_train = params['train']

    # if not os.path.exists(f"{options.out_dir}/enhancer-gene-pairs2_predictions.tsv"):
    seqnn_model = seqnn.SeqNN(params_model)
    seqnn_model.restore(model_file, 0)
    seqnn_model.build_slice(None)
    seqnn_model.build_ensemble(options.rc, options.shifts)

    enhancer_gene_pairs = pd.read_csv("/global/scratch/users/poojakathail/basenji2/enhancer_gene_benchmark_data/fig2-enhancer-gene-pairs2.tsv", sep="\t", index_col=0)
    enhancer_gene_pairs["enhancer_mid"] = enhancer_gene_pairs["enhancer_start"] + (enhancer_gene_pairs["enhancer_end"]-enhancer_gene_pairs["enhancer_start"])/2
    enhancer_gene_pairs = enhancer_gene_pairs[enhancer_gene_pairs["relative_main_tss_distance"].abs() <= 27_000]
    enhancer_gene_pairs.index = np.arange(len(enhancer_gene_pairs))

    fasta_open = pysam.FastaFile(options.genome_fasta)
    phylop_bw = pyBigWig.open(options.phylop_bigwig)
    phylop_mean = phylop_bw.header()["sumData"]/phylop_bw.header()["nBasesCovered"]
    
    # K562 CAGE targets
    # 5111, 4828
    k562_cage_target = 5111

    for i, row in tqdm(enhancer_gene_pairs.iterrows(), total=len(enhancer_gene_pairs)):
        try:
            seq, shuffled_seqs = construct_reference_sequence(row, fasta_open=fasta_open, 
                                                             phylop_bw=phylop_bw, n=options.n, phylop=params_train.get("phylop", False),
                                                             phylop_smooth=options.phylop_smooth)
            stacked_seqs = np.stack([seq] + shuffled_seqs)
            print(stacked_seqs.shape)
            preds = seqnn_model.predict(stacked_seqs)
            preds = preds[:,447:450,k562_cage_target].sum(axis=1)
            enhancer_gene_pairs.loc[i, ["Ref seq pred"] + [f"Shuffled seq {j} pred" for j in range(options.n)]] = preds
        except Exception as e:
            print(i, e)
            continue
        
    enhancer_gene_pairs.to_csv(f"{options.out_dir}/enhancer-gene-pairs2_predictions.tsv",
                               sep="\t", index=True, header=True)

    
    enhancer_gene_pairs = pd.read_csv(f"{options.out_dir}/enhancer-gene-pairs2_predictions.tsv",
                                      sep="\t", index_col=0)
    bins = [(0, 27_000), (0, 3_000), (3_000, 12_500), (12_500, 27_000)]
    results = {}
    for dataset in enhancer_gene_pairs["dataset_name"].unique():
        results_df = pd.DataFrame([])
        for bin_start, bin_end in bins:
            df_subs = enhancer_gene_pairs[(enhancer_gene_pairs["dataset_name"] == dataset) &
                        (enhancer_gene_pairs["relative_main_tss_distance"].abs() >= bin_start) &
                        (enhancer_gene_pairs["relative_main_tss_distance"].abs() < bin_end)]
            df_subs = df_subs.dropna(subset=["Ref seq pred", "Shuffled seq 0 pred"])
            
            results_df.loc["Number of positive examples", f"{bin_start}-{bin_end}"] = len(df_subs[df_subs["validated"] == True])
            results_df.loc["Number of negative examples", f"{bin_start}-{bin_end}"] = len(df_subs[df_subs["validated"] == False])
            for i in range(options.n):
                score = (df_subs["Ref seq pred"] - df_subs[F"Shuffled seq {i} pred"]).abs().values
                results_df.loc[f"AUPRC {i}", f"{bin_start}-{bin_end}"] = average_precision_score(df_subs["validated"].values, score)
        results_df.to_csv(f"{options.out_dir}/enhancer-gene-pairs_{dataset}_results_summary.tsv",
                          sep="\t", header=True, index=True)


def construct_reference_sequence(row, fasta_open, phylop_bw, n=1, phylop=False, phylop_smooth=[1], phylop_mask=True):
    seq_chr = row["chromosome"]
    tss_start = row["main_tss_start"]
    seq_start = tss_start - 131_072//2
    seq_end = tss_start + 131_072//2

    enhancer_start = int(row["enhancer_mid"] - seq_start - 1_000)
    enhancer_end = int(enhancer_start + 2_000)
    
    ref_seq_str = fasta_open.fetch(seq_chr, seq_start, seq_end)
    enhancer_seq_list = list(ref_seq_str[enhancer_start:enhancer_end])
    seq_with_shuffled_enhancer_str = []
    for i in range(n):
        random.seed(i)
        random.shuffle(enhancer_seq_list)
        enhancer_seq_shuffled = ''.join(enhancer_seq_list)
        seq_with_shuffled_enhancer_str.append(ref_seq_str[:enhancer_start] + enhancer_seq_shuffled + ref_seq_str[enhancer_end:])
    
    ref_seq = dna_io.dna_1hot(ref_seq_str)
    seq_with_shuffled_enhancer = []
    for i in range(n):
        seq_with_shuffled_enhancer.append(dna_io.dna_1hot(seq_with_shuffled_enhancer_str[i]))
        
    if phylop:
        phylop_mean = phylop_bw.header()["sumData"]/phylop_bw.header()["nBasesCovered"]
        # phylop_seq = np.array(phylop_bw.values(seq_chr, seq_start, seq_end))
        # phylop_seq[np.isnan(phylop_seq)] = phylop_mean
        seq_phylop = []
        for phylop_smooth_val in phylop_smooth:
          seq_phylop_i = np.array(phylop_bw.values(seq_chr, seq_start - phylop_smooth_val//2, seq_end + phylop_smooth_val//2))
          seq_phylop_i[np.isnan(seq_phylop_i)] = phylop_mean
          if phylop_smooth_val > 1:
            seq_phylop_i = np.convolve(seq_phylop_i, np.ones(phylop_smooth_val+1)/(phylop_smooth_val+1), mode="valid")
          seq_phylop_i = seq_phylop_i.reshape(-1, 1)
          seq_phylop.append(seq_phylop_i)
        seq_phylop = np.hstack(seq_phylop)
        
        phylop_seq_with_shuffled_enhancer = deepcopy(seq_phylop)
        if phylop_mask:
            phylop_seq_with_shuffled_enhancer[enhancer_start:enhancer_end] = phylop_mean
        
        # ref_seq = np.concatenate([ref_seq, phylop_seq.reshape(-1,1)], axis=1)
        ref_seq = np.hstack([ref_seq, seq_phylop])
        seq_with_shuffled_enhancer_final = []
        for i in range(n):
           # seq_with_shuffled_enhancer_final.append(np.concatenate([seq_with_shuffled_enhancer[i], 
           #                                                         phylop_seq_with_shuffled_enhancer.reshape(-1,1)], axis=1))
            seq_with_shuffled_enhancer_final.append(np.hstack([seq_with_shuffled_enhancer[i], 
                                                                phylop_seq_with_shuffled_enhancer]))
        return ref_seq, seq_with_shuffled_enhancer_final
    else:
        return ref_seq, seq_with_shuffled_enhancer

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()