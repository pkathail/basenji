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

from collections import OrderedDict
import pdb
import sys

import h5py
import numpy as np
import pandas as pd

import basenji.dna_io
from basenji.gene import TSS, GeneSeq

class GeneData:
    def __init__(self, genes_hdf5_file, worker_index=None, workers=None):
        # open HDF5
        self.genes_hdf5_in = h5py.File(genes_hdf5_file)

        # simple stats
        self.num_seqs, self.seq_length, self.seq_depth = self.genes_hdf5_in['seqs_1hot'].shape
        self.pool_width = int(np.array(self.genes_hdf5_in['pool_width']))

        #########################################
        # gene sequences

        self.gene_seqs = []
        for si in range(len(self.genes_hdf5_in['seq_chrom'])):
            gene_seq = GeneSeq(self.genes_hdf5_in['seq_chrom'][si].decode('UTF-8'),
                               self.genes_hdf5_in['seq_start'][si],
                               self.genes_hdf5_in['seq_end'][si])
            self.gene_seqs.append(gene_seq)

        self.seqs_1hot = np.array(self.genes_hdf5_in['seqs_1hot'])

        #########################################
        # TSS information

        self.tss = []

        for tss_i in range(len(self.genes_hdf5_in['tss_id'])):
            # map to gene seq
            seq_i = self.genes_hdf5_in['tss_seq'][tss_i]
            tss_seq = self.gene_seqs[seq_i]

            # read in TSS
            tss = TSS(self.genes_hdf5_in['tss_id'][tss_i].decode('UTF-8'),
                      self.genes_hdf5_in['tss_gene'][tss_i].decode('UTF-8'),
                      self.genes_hdf5_in['tss_chrom'][tss_i].decode('UTF-8'),
                      self.genes_hdf5_in['tss_pos'][tss_i],
                      tss_seq)
                      # tss_seq,
                      # self.genes_hdf5_in['tss_strand'][tss_i].decode('UTF-8'))
            self.tss.append(tss)

            # append to GeneSeq
            tss_seq.tss_list.append(tss)


        # do I need an ordered gene list?


        #########################################
        # determine genes split across sequences

        '''
        gene_seqs = {}
        for seq_i in range(len(self.seq_coords)):
            for transcript, tx_pos in self.seq_transcripts[seq_i]:
                gene = self.transcript_genes[transcript]
                gene_seqs.setdefault(gene,set()).add(seq_i)

        self.multi_seq_genes = set()
        for gene in gene_seqs:
            if len(gene_seqs[gene]) > 1:
                self.multi_seq_genes.add(gene)
        '''


        #########################################
        # target information

        if 'tss_targets' in self.genes_hdf5_in:
            self.tss_targets = self.genes_hdf5_in['tss_targets']
            self.target_labels = [tl.decode('UTF-8') for tl in self.genes_hdf5_in['target_labels']]
            if 'target_ids' in self.genes_hdf5_in:   # TEMP
                self.target_ids = [tl.decode('UTF-8') for tl in self.genes_hdf5_in['target_ids']]
            else:
                self.target_ids = ['']*len(self.target_labels)
            self.num_targets = len(self.target_labels)

        else:
            self.tss_targets = None
            self.target_ids = None
            self.target_labels = None
            self.num_targets = None


    def subset_genes(self, gene_ids):
        ''' Limit the sequences to a subset containing the given transcripts. '''

        if type(gene_ids) != set:
            gene_ids = set(gene_ids)

        seq_mask = np.zeros(self.num_seqs, dtype='bool')
        tss_mask = []
        for si in range(self.num_seqs):
            # determine TSSs matching given genes.
            seq_tss_list = [tss for tss in self.gene_seqs[si].tss_list if tss.gene_id in gene_ids]
            seq_tss_mask = [tss.gene_id in gene_ids for tss in self.gene_seqs[si].tss_list]
            tss_mask += seq_tss_mask

            # filter TSSs to those matching given genes.
            if len(seq_tss_list) > 0:
                seq_mask[si] = True
                self.gene_seqs[si].tss_list = seq_tss_list

        # filter sequences for those with a match
        self.gene_seqs = [self.gene_seqs[si] for si in range(self.num_seqs) if seq_mask[si]]
        self.seqs_1hot = self.seqs_1hot[seq_mask]
        self.num_seqs = len(self.gene_seqs)

        if self.tss_targets is not None:
            tss_mask = np.array(tss_mask,dtype='bool')
            print('tss_mask', tss_mask.shape)
            print('tss_targets', self.tss_targets.shape)
            self.tss_targets = self.tss_targets[tss_mask,:]


    # def subset_transcripts(self, transcripts):
    #     ''' Limit the sequences to a subset containing the given transcripts. '''

    #     seq_mask = np.zeros(self.num_seqs, dtype='bool')
    #     for si in range(self.num_seqs):
    #         # check this sequence's transcripts for matches
    #         seq_si_mask = [tx_id in transcripts for tx_id, tx_pos in self.seq_transcripts[si]]

    #         # if some transcripts match
    #         if np.sum(seq_si_mask) > 0:
    #             # keep the sequence
    #             seq_mask[si] = True

    #             # filter the transcript list
    #             self.seq_transcripts[si] = [self.seq_transcripts[si][sti] for sti in range(len(seq_si_mask)) if seq_si_mask[sti]]

    #     # filter the sequence data structures
    #     self.seq_coords = [self.seq_coords[si] for si in range(self.num_seqs) if seq_mask[si]]
    #     self.seqs_1hot = self.seqs_1hot[seq_mask,:,:]
    #     self.seq_transcripts = [self.seq_transcripts[si] for si in range(self.num_seqs) if seq_mask[si]]
    #     self.num_seqs = len(self.seq_coords)

    #     # transcript_map will point to the wrong sequences


    # def worker(self, wi, worker_num):
    #     ''' Limit the sequences to one worker's share. '''

    #     worker_mask = np.array([si % worker_num == wi for si in range(self.num_seqs)])

    #     self.seqs_1hot = self.seqs_1hot[worker_mask,:,:]
    #     self.seq_coords = [self.seq_coords[si] for si in range(self.num_seqs) if worker_mask[si]]
    #     self.seq_transcripts = [self.seq_transcripts[si] for si in range(self.num_seqs) if worker_mask[si]]
    #     self.num_seqs = len(self.seq_coords)

        # transcript_map will point to the wrong sequences

    def gene_ids(self):
        return list(self.gene_tss().keys())


    def gene_tss(self):
        gene_tss = OrderedDict()
        for tss_i in range(len(self.tss)):
            gene_tss.setdefault(self.tss[tss_i].gene_id,[]).append(tss_i)
        return gene_tss


    def tss_ids(self):
        return [tss.identifier for tss in self.tss]


    def __exit__(self):
        # close HDF5
        self.genes_hdf5_in.close()
