from scipy.stats import beta as sci_beta
from scipy.stats import dirichlet, multinomial
from scipy.special import gammaln, xlogy
import time, os, gzip, shutil, pickle, random, sys
import numpy as np
import arviz as az
from numba import jit
import pandas as pd
from os import listdir
import networkx as nx
from copy import deepcopy
from collections import Counter
from itertools import combinations
import numba as nb
import zipfile


def computeDF(n_sample,effective_k,n_introns,result_df,gene_name,z_matrix,starts,ends):
    counter = 0
    for sample_id in range(n_sample):
        for cl in range(effective_k):
            for intr in range(n_introns):
                # start = int(id2w[intr].split('-')[0])
                # end = int(id2w[intr].split('-')[1])
                result_df.iloc[counter, :] = gene_name, cl, intr, starts[intr], ends[intr], sample_id, z_matrix[
                    sample_id, intr, cl]
                counter += 1

def computeDF_vectorized(n_sample,effective_k,n_introns,result_df,gene_name,z_matrix,starts,ends):
    cls, intrs, startss, endss, sample_ids, zs = getvecs(n_sample*effective_k*n_introns,n_sample,effective_k,
                                                         n_introns,starts,ends,z_matrix)
    result_df.gene = gene_name
    result_df.trans_id = cls
    result_df["index"] = intrs
    result_df.start = startss
    result_df.end = endss
    result_df["sample"] = sample_ids
    result_df.FPKM = zs

@nb.jit(nb.types.UniTuple(nb.int32[:],6)(nb.int32,nb.int32,nb.int32,nb.int32,nb.int32[:],nb.int32[:],nb.int32[:,:,:]),nopython=True)
def getvecs(overallsize,n_sample,effective_k,n_introns,starts,ends,z_matrix):
    cls = np.zeros(overallsize, dtype=np.int32)
    intrs = np.zeros(overallsize, dtype=np.int32)
    startss = np.zeros(overallsize, dtype=np.int32)
    endss = np.zeros(overallsize, dtype=np.int32)
    sample_ids = np.zeros(overallsize, dtype=np.int32)
    zs = np.zeros(overallsize, dtype=np.int32)
    idx = 0
    for sample_id in range(n_sample):
        for cl in range(effective_k):
            for intr in range(n_introns):
                cls[idx]=cl
                intrs[idx]=intr
                startss[idx]=starts[intr]
                endss[idx]=ends[intr]
                sample_ids[idx]=sample_id
                zs[idx]=z_matrix[sample_id, intr, cl]
                idx+=1
    return cls,intrs,startss,endss,sample_ids,zs

def compress_and_delete(jsonfilename):
    if os.path.exists(jsonfilename):
        with open(jsonfilename, 'rb') as f_in:
            with gzip.open(jsonfilename + '.gz', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(jsonfilename)


# def write_split(txt_add, gene_list):
#     for i in range(len(gene_list) // 100):
#         txt_file = txt_add + str(i)
#         this_gene_list = gene_list[i * 100:(i + 1) * 100]
#         list2txt(txt_file, this_gene_list)
#
#     last_gene_list = gene_list[(i + 1) * 100:]
#     list2txt(txt_add + str(i + 1), last_gene_list)

#
# def list2txt(file_name, gene_list):
#     with open(file_name + ".txt", "w") as fobj:
#         for x in gene_list[:-1]:
#             fobj.write(x + "\n")
#         fobj.write(gene_list[-1])
#

def get_lo(intersection_M):
    lo = np.zeros([intersection_M.shape[0], 1], dtype=int)
    # compute lo
    for node in range(intersection_M.shape[0]):

        lo_set = []
        all_adj = np.where(intersection_M[node, :] == 1)[0]
        for adj in all_adj:
            if adj < node:
                lo_set.append(adj)

        if len(lo_set) == 0:
            lo_set.append(node)

        lo[node] = min(lo_set)
    return lo


def generalized_min_node_cover(intersection_M, i=2):
    lo = get_lo(intersection_M)
    W = np.zeros([intersection_M.shape[0], 1], dtype=int)
    MVC = []

    for node in range(intersection_M.shape[0]):
        must = False
        for u in range(int(lo[node]), node + 1):
            W[u] += 1
            if W[u] == i:
                must = True
        if must == True:
            MVC.append(node)
            for u in range(int(lo[node]), node + 1):
                W[u] -= 1
    return MVC


def find_min_clusters(nodes_df):
    _, edges_list = get_conflict_for_plot(nodes_df)
    G = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    min_k = nx.graph_clique_number(G, cliques=None)
    # min_k = len(nx.maximal_independent_set(G))
    return min_k


def get_conflict_for_plot(nodes_df):
    """Find the intervals that have intersection"""
    intersection_M = np.zeros([nodes_df.shape[0], nodes_df.shape[0]], dtype=int)
    edges_list = []
    for v1 in range(nodes_df.shape[0]):
        s1 = nodes_df.loc[v1, 'start']
        e1 = nodes_df.loc[v1, 'end']
        for v2 in range(v1 + 1, nodes_df.shape[0]):
            s2 = nodes_df.loc[v2, 'start']
            e2 = nodes_df.loc[v2, 'end']
            if e1 > s2 and s1 < e2:
                intersection_M[v1, v2] = 1
                intersection_M[v2, v1] = 1
                edges_list.append((v1, v2))
    return intersection_M, edges_list


def generate_interval_graph_nx(nodes_df, edges_list, intervalviz=True):
    """Generate the graph G=(V,E) using networkx library and visualize"""
    G = nx.Graph()
    if intervalviz:
        newedgesList = [(nodes_df['graph_labels'][ee[0]], nodes_df['graph_labels'][ee[1]]) for ee in edges_list]
        G.add_nodes_from(nodes_df['graph_labels'])
    else:
        newedgesList = [(nodes_df['node_labels'][ee[0]], nodes_df['node_labels'][ee[1]]) for ee in edges_list]
        G.add_nodes_from(nodes_df['node_labels'])
        # newedgesList = edges_list

    for e in newedgesList:
        G.add_edge(*e)
    # nx.draw_networkx(G, pos=None, arrows=False, with_labels=True, node_size= 50)
    # plt.show() # display
    return G


def split_training_test(document_orig, tr_percentage=95):
    # split training and test
    # tr_percentage = 95
    tr_size = int(tr_percentage / 100 * document_orig.shape[0])
    indices = np.random.RandomState(seed=2021).permutation(document_orig.shape[0])
    training_idx, test_idx = indices[:tr_size], indices[tr_size:]
    document = document_orig[training_idx, :]
    document_te = document_orig[test_idx, :]
    return document, document_te, training_idx, test_idx


def find_mis(nodes_df):
    _, edges_list = get_conflict_for_plot(nodes_df)
    G = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    GC = nx.complement(G)
    MIS = nx.graph_clique_number(GC, cliques=None)
    max_ind_set = nx.maximal_independent_set(G)
    while len(max_ind_set) < MIS:
        max_ind_set = nx.maximal_independent_set(G)
    max_ind_set = [int(n) for n in max_ind_set]
    max_ind_set.sort()
    return MIS, max_ind_set


def get_initialization(nodes_df, N_K):
    _, edges_list = get_conflict_for_plot(nodes_df)
    G = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    all_max_ind_set = []
    while len(all_max_ind_set) < N_K:
        temp = nx.maximal_independent_set(G)
        if temp not in all_max_ind_set:
            all_max_ind_set.append(temp)
    return all_max_ind_set


def find_initial_nodes(nodes_df, N_K):
    _, edges_list = get_conflict_for_plot(nodes_df)
    G = generate_interval_graph_nx(nodes_df, edges_list, intervalviz=False)
    all_max_ind_set = []
    for i in range(1000):
        temp = nx.maximal_independent_set(G)
        if temp not in all_max_ind_set:
            all_max_ind_set.append(temp)

    while len(all_max_ind_set) < N_K:
        temp = nx.maximal_independent_set(G)
        all_max_ind_set.append(temp)
    return all_max_ind_set


def add_node_IS_Beta(S, gene_intersection, N_V, bet):
    # add:
    free = set(range(N_V)) - set(S)
    for ss in S:
        neighbor_ss = set(np.where(gene_intersection[ss, :] == 1)[0])
        free = free - neighbor_ss
        if len(free) == 0:
            return []
    add_node = random.choices(list(free), weights=bet[list(free)] / np.sum(bet[list(free)]), k=1)
    return add_node


def del_node_IS_Beta(S, bet):
    if len(S) == 0:
        return []
    del_node = random.choices(list(S), weights=1 - (bet[S] / np.sum(bet[S])), k=1)
    return del_node


def sample_local_ind_set(gene_intersection, N_V, N_S, b_k, Beta_k, MIS):
    max_trial = 200

    S = list(np.where(b_k)[0])
    S.sort()
    random_clusters = []
    temp2 = deepcopy(S)
    random_clusters.append(temp2)
    trial = 0
    while len(random_clusters) < N_S and trial < max_trial:
        trial += 1
        # add or remove:
        # rnd = np.random.random()
        rnd = np.random.binomial(n=1, p=1 - (len(S) / MIS))
        if rnd >= 0.5:
            an = add_node_IS_Beta(S, gene_intersection, N_V, Beta_k)
            if len(an) != 0:
                S.append(an[0])
                S.sort()
                temp = deepcopy(S)
                if temp not in random_clusters:
                    random_clusters.append(temp)
        else:
            dn = del_node_IS_Beta(S, Beta_k)

            if len(dn) != 0 and len(S) != 1:
                S.remove(dn[0])
                S.sort()
                temp = deepcopy(S)
                if temp not in random_clusters and len(temp) > 0:
                    random_clusters.append(temp)

    return random_clusters


def find_duplicate_clusters(b):

    input = map(tuple, b)

    freqDict = Counter(input)

    duplicated_clusters = [row for row in freqDict.keys() if freqDict[row]>1]

    return duplicated_clusters


def merge_suplicate_clusters(b, z):
    dup_cl = find_duplicate_clusters(b)

    while len(dup_cl) > 0:
        print('hit:', len(dup_cl))
        dup0 = np.array(dup_cl[0])
        dup0_indices = list(np.where(np.all(b == dup0, axis=1))[0])
        removing_indices = dup0_indices[1:]

        for dd in removing_indices[::-1]:
            b = np.delete(b, dd, 0)

        z[:, :, dup0_indices[0]] = np.sum(z[:, :, dup0_indices], axis=2)

        for dd in removing_indices[::-1]:
            z = np.delete(z, dd, 2)

        dup_cl = find_duplicate_clusters(b)
    return b, z


def save_results(gene, model):
    print('Saving the results for gene', gene.name)
    comb_name = 'gene_' + gene.name + '_alpha_' + str(model.alpha) + '_eta_' + str(model.eta) + '_epsilon_' + \
                str(model.epsilon) + '_rs_' + str(model.r) + '_K_' + str(model.run_info['N_K'])
    last_run = list(model.run_info['gibbs'])[-1]
    last_z = deepcopy(model.run_info['gibbs'][last_run]['Z'])
    last_b = deepcopy(model.run_info['gibbs'][last_run]['b'])
    new_b, new_z = merge_suplicate_clusters(last_b, last_z)
    model.run_info['new_b'] = deepcopy(new_b)
    model.run_info['new_Z'] = deepcopy(new_z)
    # save the result
    if not os.path.exists(gene.result_path):
        os.mkdir(gene.result_path)
    # pickle.dump(model.run_info, open(gene.result_path + '/' + 'run_info_' + comb_name + '.json', 'wb'))
    filename = gene.result_path + '/' + 'run_info_' + comb_name + '.pkl'
    file_s = gzip.GzipFile(filename, 'wb')
    pickle.dump(model.run_info, file_s)
    print(filename, 'saved.')

    z_matrix = model.run_info['new_Z']
    id2w = model.run_info['id2w_dict']
    n_sample = z_matrix.shape[0]
    n_introns = z_matrix.shape[1]
    effective_k = z_matrix.shape[2]
    gene_name = model.run_info['gene']


    starts = np.asarray([int(id2w[j].split('-')[0]) for j in range(n_introns)], np.int32)
    ends = np.asarray([int(id2w[j].split('-')[1]) for j in range(n_introns)], np.int32)

    result_df = pd.DataFrame(data=0, columns=['gene', 'trans_id', 'index', 'start', 'end', 'sample', 'FPKM'],
                             index=range(n_sample * effective_k * n_introns))

    # result_df2 = pd.DataFrame(data=0, columns=['gene', 'trans_id', 'index', 'start', 'end', 'sample', 'FPKM'],
    #                          index=range(n_sample*effective_k*n_introns))

    computeDF_vectorized(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends)

    # computeDF(n_sample,effective_k,n_introns,result_df2,gene_name,z_matrix,starts,ends)

    file_name_2 = 'bamie_' + gene_name + '_K_' + str(effective_k) + '.csv'
    result_df.to_csv(gene.result_path + '/' + file_name_2)
    print(gene.result_path + '/' + file_name_2, 'saved.')
    return gene.result_path + '/' + file_name_2


def needed_n_k_list(gene):
    if os.path.exists(gene.result_path):
        # done_comb = [dname.split('run_info_')[1].split('.json')[0] for dname in os.listdir(gene.result_path) if
        # '.json' in dname]
        # done_k = [int(comb.split('.json')[0].split('_K_')[1]) for comb in done_comb]
        done_k = []
    else:
        done_k = []
    N_K_v = sorted(gene.all_n_k[::2][:9])
    N_K_v = list(set(N_K_v) - set(done_k))
    return N_K_v


def make_confusion_matrix(gene_run_info, tr_ex_int_df, this_gene, threshold, cut_off_threshold):

    # cut_off_threshold, distance_threshold = 100, 100
    This_gene_tr_int_df = tr_ex_int_df[tr_ex_int_df['gene'] == this_gene].reset_index()
    This_gene_tr = list(set(This_gene_tr_int_df['transcript_id']))
    This_gene_tr_int_df['tr_idx'] = -1
    This_gene_tr_int_df['intron_idx'] = -1

    # index transcripts
    for tt in range(len(This_gene_tr_int_df)):
        This_gene_tr_int_df.loc[tt, 'tr_idx'] = int(np.where(np.array(This_gene_tr) == This_gene_tr_int_df.loc[tt, 'transcript_id'])[0])

    # index the intron excisions
    param_dict = gene_run_info['param_dict']
    dictw2id = param_dict['dictw2id']
    for dw in dictw2id.keys():
        start = int(dw.split('-')[0])
        end = int(dw.split('-')[1])
        start_match_list = list(np.abs(This_gene_tr_int_df['start']-start) <= 6)
        end_match_list = list(np.abs(This_gene_tr_int_df['end'] - end) <= 6)
        same_int_list = [a and b for a, b in zip(start_match_list, end_match_list)]
        same_int_idx = [i for i, x in enumerate(same_int_list) if x]
        This_gene_tr_int_df.loc[same_int_idx, 'intron_idx'] = dictw2id[dw]

    This_gene_tr_int_df = This_gene_tr_int_df[This_gene_tr_int_df['intron_idx'] != -1]
    This_gene_tr_int_df = This_gene_tr_int_df.sort_values(['tr_idx', 'intron_idx'], ascending=True)
    this_gene_df_agg = This_gene_tr_int_df.sort_values(['tr_idx', 'intron_idx'], ascending=True).groupby(['tr_idx'])['intron_idx'].apply(list)

    if 'new_b' not in gene_run_info.keys():

        last_run = list(gene_run_info['gibbs'])[-1]
        last_z = deepcopy(gene_run_info['gibbs'][last_run]['Z'])
        last_b = deepcopy(gene_run_info['gibbs'][last_run]['b'])
        new_b, new_z = merge_suplicate_clusters(last_b, last_z)

    else:
        new_b = deepcopy(gene_run_info['new_b'])
        new_z = deepcopy(gene_run_info['new_Z'])

    A = sorted(list(set(This_gene_tr_int_df['intron_idx'])))
    all_pairs = list(combinations(A, 2))

    b_clusters = set()
    for kk in range(new_b.shape[0]):
        this_kk = list(np.where(new_b[kk, :])[0])
        this_kk = sorted(this_kk)
        this_kk_pairs = list(combinations(this_kk, 2))
        b_clusters = b_clusters.union(set(this_kk_pairs))
    b_clusters = sorted(list(b_clusters))

    expressed_transcripts = []
    not_expressed_transcripts = []
    existing_transcripts = sorted(list(set(This_gene_tr_int_df['tr_idx'])))

    for i in existing_transcripts:

    # for i in range(len(this_gene_df_agg)):

        if len(list(np.where(np.sum(np.sum(new_z[:, this_gene_df_agg[i], :], axis=1), axis=1) > threshold)[
                        0])) > cut_off_threshold:
            expressed_transcripts.append(i)
        else:
            not_expressed_transcripts.append(i)

    expressed_tr_pairs = set()
    for ex_tr in expressed_transcripts:
        expressed_tr_pairs = expressed_tr_pairs.union(set(combinations(this_gene_df_agg[ex_tr], 2)))
    expressed_tr_pairs = sorted(list(expressed_tr_pairs))

    TP_pairs = []
    TN_pairs = []
    FP_pairs = []
    FN_pairs = []

    for pair in all_pairs:
        if pair in b_clusters and pair in expressed_tr_pairs:
            TP_pairs.append(pair)
        elif pair in b_clusters and pair not in expressed_tr_pairs:
            FP_pairs.append(pair)
        elif pair not in b_clusters and pair in expressed_tr_pairs:
            FN_pairs.append(pair)
        else:
            TN_pairs.append(pair)

    TN = len(TN_pairs)
    TP = len(TP_pairs)
    FN = len(FN_pairs)
    FP = len(FP_pairs)

    if TP + FP == 0:
        precision = np.nan
    else:
        precision = TP/(TP+FP)

    if TP + FN == 0:
        recall = np.nan
    else:
        recall = TP/(TP+FN)

    return TN, TP, FN, FP, precision, recall


def compute_config_score(sam_df, trans_introns_f, config):
    # config = [1, 3, 7]
    tr_score_list = []
    for trii in trans_introns_f:
        this_tr_score = 0
        for node in config:
            node_interval = list(sam_df.iloc[node, :].values)
            for intr in trans_introns_f[trii]:
                intr_start = trans_introns_f[trii][intr]['start']
                intr_end = trans_introns_f[trii][intr]['end']
                intron_interval = [intr_start, intr_end]

                if np.abs(node_interval[0] - intron_interval[0]) <= 12 and np.abs(
                        node_interval[1] - intron_interval[1]) <= 12:
                    this_tr_score += 1
                    break
                # else:
                #     print('start points difference', np.abs(node_interval[0] - intron_interval[0]),
                #           'end points difference', np.abs(node_interval[1] - intron_interval[1]))
        tr_score_list.append(this_tr_score)
    if len(tr_score_list) !=  0:
        this_config_score = max(tr_score_list) / len(config)
    else:
        this_config_score = 0
    return this_config_score


def calc_bic(n_d, n_v, n_k, max_l):
    bic_k = n_v + n_k
    return (bic_k * np.log(n_d)) - (2 * max_l)


def calc_bic2(n_d, n_v, n_k, all_n_w, max_l):
    num_theta = n_k - 1
    num_z = all_n_w * (n_k - 1)
    num_beta = n_v * (n_k - 1)
    num_b = n_v * n_k
    num_pi = n_k - 1

    bic_k = num_theta + num_z + num_beta + num_b + num_pi
    return (bic_k * np.log(n_d)) - (2 * max_l)


def gene_best_comb(main_folder, this_gene):

    results_name = '_results.csv'

    grid_df = pd.read_csv(main_folder + '/all_results/' + this_gene + '/' + this_gene + results_name)
    if len(grid_df) > 0:
        grid_agg_df = grid_df.groupby(['gene', 'N_K', 'alpha', 'eta', 'rs', 'epsilon', 'N_V', 'Max_likelihood', 'BIC1',
                                       'BIC2', 'predictive_likelihood']).agg({'config': 'count', 'score': 'mean'}) \
            .rename(columns={'config': 'n_clusters', 'score': 'score_avg', 'BIC1': 'BIC'}).reset_index()

        grid_agg_df = grid_agg_df.rename(columns={'BIC1': 'BIC'}, errors="raise")

        target_col_name = 'N_K'
        n_k_list = list(set(grid_agg_df[target_col_name]))
        pred_L_max = []
        max_likelihood_list_max = []
        best_tr_df = pd.DataFrame(columns=['gene', 'N_K', 'alpha', 'eta', 'rs', 'epsilon', 'str', 'Max_likelihood'],
                                  index=n_k_list)
        best_te_df = pd.DataFrame(columns=['gene', 'N_K', 'alpha', 'eta', 'rs', 'epsilon', 'str', 'predictive_likelihood'],
                                  index=n_k_list)

        for N_K in n_k_list:
            this_df = grid_agg_df[grid_agg_df[target_col_name] == N_K]

            max_likelihood_list_max.append(np.max(this_df['Max_likelihood']))
            pred_L_max.append(np.max(this_df['predictive_likelihood']))

            max_tr = np.max(this_df['Max_likelihood'])
            max_te = np.max(this_df['predictive_likelihood'])

            # training
            best_comb_tr = this_df[this_df['Max_likelihood'] == max_tr]
            best_alpha_tr = best_comb_tr['alpha'].values[0]
            best_eta_tr = best_comb_tr['eta'].values[0]
            best_rs_tr = best_comb_tr['rs'].values[0]
            best_epsilon_tr = best_comb_tr['epsilon'].values[0]
            best_alpha_tr_str = str(best_alpha_tr) if best_alpha_tr < 1 else str(int(best_alpha_tr))
            best_eta_tr_str = str(best_eta_tr) if best_eta_tr < 1 else str(int(best_eta_tr))
            best_rs_tr_str = str(best_rs_tr) if best_rs_tr < 1 else str(int(best_rs_tr))
            best_epsilon_tr_str = str(best_epsilon_tr) if best_epsilon_tr < 1 else str(int(best_epsilon_tr))
            best_tr_str = 'run_info_gene_' + this_gene + '_alpha_' + best_alpha_tr_str + '_eta_' + best_eta_tr_str + \
                          '_epsilon_' + best_epsilon_tr_str + '_rs_' + best_rs_tr_str + '_K_' + str(N_K) + '.json'
            best_tr_df.loc[N_K, :] = this_gene, N_K, best_alpha_tr, best_eta_tr, best_rs_tr, best_epsilon_tr, best_tr_str, \
                                     max_tr

            # test
            best_comb_te = this_df[this_df['predictive_likelihood'] == max_te]
            best_alpha_te = best_comb_te['alpha'].values[0]
            best_eta_te = best_comb_te['eta'].values[0]
            best_rs_te = best_comb_te['rs'].values[0]
            best_epsilon_te = best_comb_te['epsilon'].values[0]
            best_alpha_te_str = str(best_alpha_te) if best_alpha_te < 1 else str(int(best_alpha_te))
            best_eta_te_str = str(best_eta_te) if best_eta_te < 1 else str(int(best_eta_te))
            best_rs_te_str = str(best_rs_te) if best_rs_te < 1 else str(int(best_rs_te))
            best_epsilon_te_str = str(best_epsilon_te) if best_epsilon_te < 1 else str(int(best_epsilon_te))
            best_te_str = 'run_info_gene_' + this_gene + '_alpha_' + best_alpha_te_str + '_eta_' + best_eta_te_str + \
                          '_epsilon_' + best_epsilon_te_str + '_rs_' + best_rs_te_str + '_K_' + str(N_K) + '.json'
            best_te_df.loc[N_K, :] = this_gene, N_K, best_alpha_te, best_eta_te, best_rs_te, best_epsilon_te, best_te_str, \
                                     max_te
        best_tr_name = '_best_comb_tr.csv'
        best_te_name = '_best_comb_te.csv'
        best_tr_df.to_csv(main_folder + '/all_results/' + this_gene + '/' + this_gene + best_tr_name)
        best_te_df.to_csv(main_folder + '/all_results/' + this_gene + '/' + this_gene + best_te_name)
    else:
        print(this_gene, 'has no results csv')
        # best_global_te = best_te_df[best_te_df['predictive_likelihood'] == max(best_te_df['predictive_likelihood'])]['str'].values[0]
        # best_global_tr = best_tr_df[best_tr_df['Max_likelihood'] == max(best_tr_df['Max_likelihood'])]['str'].values[0]
        # return best_tr_name, best_te_name, best_global_tr, best_global_te


@jit(nopython=True)
def adjust_matrices(mat, eps):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] < eps:
                mat[i, j] = eps
    return mat


@jit
def update_z_loop_numba(beta, theta, n_tr, n_v, n_k, document_tr):
    z = np.array([n_tr, n_v, n_k])
    for doc in range(0, n_tr):
        for v in range(0, n_v):
            ratio_v = np.exp(np.log(theta[doc, :]) + np.log(beta[:, v]))
            ratio_v /= np.sum(ratio_v)
            tempz = np.random.multinomial(1, ratio_v, size=document_tr[doc, v]).argmax(axis=1)
            for k in range(0, n_k):
                z[doc, v, k] = np.count_nonzero(tempz == k)
    return z


def read_run_info(path):
    if os.path.getsize(path) == 0:
        run_info = 0

    else:
        if '.gz' in path:
            with gzip.open(path) as handle:
                run_info = pickle.load(handle)
        elif '.json' in path and os.path.getsize(path) > 0:
            with open(path, 'rb') as handle:
                run_info = pickle.load(handle)

        elif '.pkl' in path:
            with gzip.open(path, 'rb') as ifp:
                run_info = pickle.load(ifp)

    return run_info


def compute_local_metrics(closest_tr, This_gene_tr_int_df, T_k):
    T_t_df = This_gene_tr_int_df[This_gene_tr_int_df['transcript_id'] == closest_tr].reset_index()

    T_t = sorted(list(set([(int(T_t_df.loc[i, 'start']), int(T_t_df.loc[i, 'end'])) for i in range(len(T_t_df))])))
    local_FP = 0
    local_TP = 0
    local_FN = 0
    for intron in T_k:

        if any([isthesameintron(true_intron, intron) for true_intron in T_t]):
            local_TP += 1
        else:
            local_FP += 1
    for tr_intron in T_t:
        if not any([isthesameintron(tr_intron, intr) for intr in T_k]):
            local_FN += 1

    precision = local_TP / (local_TP + local_FP)
    recall = local_TP / (local_TP + local_FN)
    if precision + recall != 0:

        F1Score = (2 * precision * recall) / (precision + recall)
    else:
        F1Score = 0
    return local_FP, local_TP, local_FN, precision, recall, F1Score


def closest_transcript_eq2(T_k, This_gene_tr_int_df, this_sam_cl_res):

    if 'FPKM' in this_sam_cl_res.columns.values:
        col_name = 'FPKM'
    else:
        col_name = 'count'

    this_cluter_introns_unique = T_k
    This_gene_tr = list(set(This_gene_tr_int_df['transcript_id']))

    phs_df = pd.DataFrame(data=0, columns=['tr_id', 'phs_score', 'tr_quant_unnorm', 'tr_quant_norm', 'n_introns'], index=range(len(This_gene_tr)))
    t_k_df = pd.DataFrame(data=0, columns=['tr_id', 'tr_intron'] + [str(k) for k in T_k], index=range(len(This_gene_tr_int_df)))

    tk_cc = 0
    tr_score_list = []
    for tr_id in range(len(This_gene_tr)):
        tr = This_gene_tr[tr_id]

        this_tr = This_gene_tr_int_df[This_gene_tr_int_df['transcript_id'] == tr]
        this_tr_size = this_tr.groupby(['start', 'end']).size().reset_index()
        this_tr_true_introns = [(int(this_tr_size.loc[i, 'start']), int(this_tr_size.loc[i, 'end'])) for i in range(len(this_tr_size))]

        counter = 0
        for intr_id in range(len(this_tr_true_introns)):
            intr = this_tr_true_introns[intr_id]
            t_k_df.loc[tk_cc, 'tr_id'] = tr
            t_k_df.loc[tk_cc, 'tr_intron'] = str(intr)

            for elem in this_cluter_introns_unique:
                if isthesameintron(intr, elem):
                    counter += 1
                    t_k_df.loc[tk_cc, str(elem)] += 1
            tk_cc += 1

        phs_score = counter/len(this_cluter_introns_unique)

        tr_score_list.append(phs_score)

        phs_df.loc[tr_id, 'tr_id'] = tr
        phs_df.loc[tr_id, 'phs_score'] = phs_score

    quant_list = np.zeros([len(This_gene_tr)])
    for elem in this_cluter_introns_unique:
        elem_start = np.float(elem[0])
        elem_end = np.float(elem[1])

        fpkm = this_sam_cl_res[((this_sam_cl_res['start'] == elem_start) & (this_sam_cl_res['end'] == elem_end))][col_name].values[0]
        if np.sum(t_k_df[str(elem)]) != 0:
            unnorm_val = fpkm/np.sum(t_k_df[str(elem)])
        else:
            unnorm_val = 0

        for tr_id in range(len(This_gene_tr)):
            tr = This_gene_tr[tr_id]
            this_elem_tr_quant = np.sum(t_k_df.loc[t_k_df['tr_id'] == tr, str(elem)]) * unnorm_val
            quant_list[tr_id] += this_elem_tr_quant
    n_introns = [len(t_k_df[t_k_df['tr_id'] == tr]) for tr in This_gene_tr]
    quant_list_norm = quant_list/n_introns
    phs_df['n_introns'] = n_introns
    phs_df['tr_quant_unnorm'] = quant_list
    phs_df['tr_quant_norm'] = quant_list_norm
    max_id = np.argmax(tr_score_list)

    return This_gene_tr[max_id], phs_df


def closest_transcript_eq3(T_k, This_gene_tr_int_df, this_sam_cl_res):

    if 'FPKM' in this_sam_cl_res.columns.values:
        col_name = 'FPKM'
    else:
        col_name = 'count'

    this_cluter_introns_unique = T_k
    This_gene_tr = list(set(This_gene_tr_int_df['transcript_id']))

    phs_df = pd.DataFrame(data=0, columns=['tr_id', 'phs_score', 'tr_quant_unnorm', 'tr_quant_norm', 'n_introns'], index=range(len(This_gene_tr)))
    t_k_df = pd.DataFrame(data=0, columns=['tr_id', 'tr_intron'] + [str(k) for k in T_k], index=range(len(This_gene_tr_int_df)))

    tk_cc = 0
    tr_score_list = []
    for tr_id in range(len(This_gene_tr)):
        tr = This_gene_tr[tr_id]

        this_tr = This_gene_tr_int_df[This_gene_tr_int_df['transcript_id'] == tr]
        this_tr_size = this_tr.groupby(['start', 'end']).size().reset_index()
        this_tr_true_introns = [(int(this_tr_size.loc[i, 'start']), int(this_tr_size.loc[i, 'end'])) for i in range(len(this_tr_size))]

        counter = 0
        for intr_id in range(len(this_tr_true_introns)):
            intr = this_tr_true_introns[intr_id]
            t_k_df.loc[tk_cc, 'tr_id'] = tr
            t_k_df.loc[tk_cc, 'tr_intron'] = str(intr)

            for elem in this_cluter_introns_unique:
                if isthesameintron(intr, elem):
                    counter += 1
                    t_k_df.loc[tk_cc, str(elem)] += 1
            tk_cc += 1

        phs_score = counter/len(this_tr_true_introns)

        tr_score_list.append(phs_score)

        phs_df.loc[tr_id, 'tr_id'] = tr
        phs_df.loc[tr_id, 'phs_score'] = phs_score

    quant_list = np.zeros([len(This_gene_tr)])
    for elem in this_cluter_introns_unique:
        elem_start = np.float(elem[0])
        elem_end = np.float(elem[1])

        fpkm = this_sam_cl_res[((this_sam_cl_res['start'] == elem_start) & (this_sam_cl_res['end'] == elem_end))][col_name].values[0]
        # unnorm_val = fpkm/np.sum(t_k_df[str(elem)])

        if np.sum(t_k_df[str(elem)]) != 0:
            unnorm_val = fpkm/np.sum(t_k_df[str(elem)])
        else:
            unnorm_val = 0

        for tr_id in range(len(This_gene_tr)):
            tr = This_gene_tr[tr_id]
            this_elem_tr_quant = np.sum(t_k_df.loc[t_k_df['tr_id'] == tr, str(elem)]) * unnorm_val
            quant_list[tr_id] += this_elem_tr_quant
    n_introns = [len(t_k_df[t_k_df['tr_id'] == tr]) for tr in This_gene_tr]
    quant_list_norm = quant_list/n_introns
    phs_df['n_introns'] = n_introns
    phs_df['tr_quant_unnorm'] = quant_list
    phs_df['tr_quant_norm'] = quant_list_norm
    max_id = np.argmax(tr_score_list)

    return This_gene_tr[max_id], phs_df


def isintronintrans(tr, intron, This_gene_tr_int_df):
    this_tr = This_gene_tr_int_df[This_gene_tr_int_df['transcript_id'] == tr]
    this_tr_size = this_tr.groupby(['start', 'end']).size().reset_index()
    this_tr_true_introns = [(int(this_tr_size.loc[i, 'start']), int(this_tr_size.loc[i, 'end'])) for i in
                            range(len(this_tr_size))]
    start = intron[0]
    end = intron[1]
    start_match_list = [np.abs(elem[0] - start) <= 6 for elem in this_tr_true_introns]
    end_match_list = [np.abs(elem[1] - end) <= 6 for elem in this_tr_true_introns]
    same_int_list = [a and b for a, b in zip(start_match_list, end_match_list)]
    same_int_idx = [i for i, x in enumerate(same_int_list) if x]
    return len(same_int_idx) > 0


def isthesameintron(int1, int2):
    s1, e1 = int1
    s2, e2 = int2
    return np.abs(s1 - s2) <= 6 and np.abs(e1 - e2) <= 6


def filter_results_by_expression(gene_result_path, threshold=10, cut_off_threshold=10):
    bamie_raw = pd.read_csv(gene_result_path, index_col=0)
    bamie_raw_no_zero = bamie_raw[bamie_raw['FPKM'] != 0]
    bamie_raw_th1_df = bamie_raw_no_zero.groupby(['sample', 'trans_id'])['FPKM'].agg('sum').reset_index()
    bamie_raw_th1_df2 = bamie_raw_th1_df[bamie_raw_th1_df['FPKM'] > threshold]
    expressed_sample_clusters = list(zip(bamie_raw_th1_df2['sample'], bamie_raw_th1_df2['trans_id']))
    bamie_raw_th1_df3 = bamie_raw_th1_df2.groupby(['trans_id']).size().reset_index()
    expressed_clusters = list(bamie_raw_th1_df3[bamie_raw_th1_df3[0] > cut_off_threshold]['trans_id'])
    final_expression = [elem for elem in expressed_sample_clusters if elem[1] in expressed_clusters]
    final_expression2 = [(i, j) for (j, i) in final_expression]
    filtered_results = bamie_raw_no_zero[bamie_raw_no_zero[['trans_id', 'sample']].apply(tuple, axis=1).isin(final_expression2)].reset_index()
    return filtered_results


def compute_gene_metrics_bamie(tr_ex_int_df, cuff_res_path, this_gene, save_path, file_pattern_name):

    This_gene_tr_int_df = tr_ex_int_df[tr_ex_int_df['gene'] == this_gene].reset_index()
    method = 'BAMIE'
    # filtering by expression
    cuff_res = filter_results_by_expression(cuff_res_path, threshold=10, cut_off_threshold=10)


    samples = sorted(list(set(cuff_res['sample'])))
    n_tr = len(set(This_gene_tr_int_df['transcript_id']))

    cuff_intr_g = cuff_res.groupby(['start', 'end']).size().reset_index()
    computed_introns = sorted(
        [(int(cuff_intr_g.loc[i, 'start']), int(cuff_intr_g.loc[i, 'end'])) for i in range(len(cuff_intr_g))])

    true_introns_g = This_gene_tr_int_df.groupby(['start', 'end']).size().reset_index()
    true_introns = sorted([(int(true_introns_g.loc[i, 'start']), int(true_introns_g.loc[i, 'end'])) for i in
                           range(len(true_introns_g))])

    matching_df = pd.DataFrame(data=0, columns=[str(i) for i in true_introns],
                               index=[str(j) for j in computed_introns])
    for tc in computed_introns:
        for tt in true_introns:
            matching_df.loc[str(tc), str(tt)] = int(isthesameintron(tc, tt))

    all_clusters = sorted(list(set(cuff_res['trans_id'])))
    transcripts = sorted(list(set(This_gene_tr_int_df['transcript_id'])))

    phs_score_df_eq2 = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)
    phs_score_df_eq3 = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)

    closest_eq2, closest_eq3 = {}, {}
    # print(all_clusters)
    for cl in all_clusters:

        clusters_2 = cuff_res[cuff_res['trans_id'] == cl]
        clusters2_g = clusters_2.groupby(['start', 'end']).size().reset_index()
        clusters_introns = sorted(
            list(set([(int(clusters2_g.loc[i, 'start']), int(clusters2_g.loc[i, 'end'])) for i in range(len(clusters2_g))])))
        trans_scores_eq2 = []
        trans_scores_eq3 = []

        for trans in transcripts:

            trans_2 = This_gene_tr_int_df[This_gene_tr_int_df['transcript_id'] == trans]
            trans_2g = trans_2.groupby(['start', 'end']).size().reset_index()
            this_trans_introns = sorted(list(set([(int(trans_2g.loc[i, 'start']), int(trans_2g.loc[i, 'end'])) for i in range(len(trans_2g))])))

            counter = 0

            for tr_intr in this_trans_introns:
                for cl_intr in clusters_introns:
                    if isthesameintron(cl_intr, tr_intr) == 1:
                        counter += 1
                        break

                    counter += matching_df.loc[str(cl_intr), str(tr_intr)]

            trans_scores_eq2.append((counter / len(clusters_introns)))
            trans_scores_eq3.append((counter / len(this_trans_introns)))
            phs_score_df_eq2.loc[cl, trans] = counter / len(clusters_introns)
            phs_score_df_eq3.loc[cl, trans] = counter / len(this_trans_introns)

        max_id_eq2 = np.argmax(trans_scores_eq2)
        max_id_eq3 = np.argmax(trans_scores_eq3)
        closest_eq2[cl] = transcripts[max_id_eq2]
        closest_eq3[cl] = transcripts[max_id_eq3]


    local_metric_df_FP = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)
    local_metric_df_TP = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)
    local_metric_df_FN = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)
    local_metric_df_precision = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)
    local_metric_df_recall = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)
    local_metric_df_f1score = pd.DataFrame(data=0, columns=transcripts, index=all_clusters)
    logical_cl_tra_eq2 = pd.DataFrame(data=-1, columns=transcripts, index=all_clusters)
    logical_cl_tra_eq3 = pd.DataFrame(data=-1, columns=transcripts, index=all_clusters)

    for cl in all_clusters:
        clusters_2 = cuff_res[cuff_res['trans_id'] == cl]
        clusters2_g = clusters_2.groupby(['start', 'end']).size().reset_index()
        clusters_introns = sorted(
            [(int(clusters2_g.loc[i, 'start']), int(clusters2_g.loc[i, 'end'])) for i in range(len(clusters2_g))])

        for trans in transcripts:
            local_FP, local_TP, local_FN, precision, recall, F1Score = compute_local_metrics(trans,
                                                                                             This_gene_tr_int_df,
                                                                                             clusters_introns)
            local_metric_df_FP.loc[cl, trans] = local_FP
            local_metric_df_TP.loc[cl, trans] = local_TP
            local_metric_df_FN.loc[cl, trans] = local_FN
            local_metric_df_precision.loc[cl, trans] = precision
            local_metric_df_recall.loc[cl, trans] = recall
            local_metric_df_f1score.loc[cl, trans] = F1Score
            logical_cl_tra_eq2.loc[cl, trans] = int(closest_eq2[cl] == trans)
            logical_cl_tra_eq3.loc[cl, trans] = int(closest_eq3[cl] == trans)


    cuff_res_df_agg = cuff_res.sort_values(['trans_id'], ascending=True).groupby(['trans_id'])['sample'].apply(set)
    sam_cl_count = np.sum([len(cuff_res_df_agg.loc[i]) for i in list(cuff_res_df_agg.index)])

    gene_phs_results = pd.DataFrame(columns=['method', 'gene', 'sample', 'cluster', 'equation', 'tr_id',
                                             'phs_score', 'quant',
                                             'quant_norm', 'FP', 'TP', 'FN', 'precision', 'recall', 'F1Score',
                                             'closest'])

    # cc = 0
    for cl in list(cuff_res_df_agg.index):
        for tra in transcripts:
            fp = local_metric_df_FP.loc[cl, tra]
            tp = local_metric_df_TP.loc[cl, tra]
            fn = local_metric_df_FN.loc[cl, tra]
            prec = local_metric_df_precision.loc[cl, tra]
            reca = local_metric_df_recall.loc[cl, tra]
            fscore = local_metric_df_f1score.loc[cl, tra]
            phs_2 = phs_score_df_eq2.loc[cl, tra]
            phs_3 = phs_score_df_eq3.loc[cl, tra]


            this_np_eq2 = np.array(
                [method, this_gene, -1, cl, 2, tra, phs_2, 0, 0, fp, tp, fn, prec, reca, fscore,
                 logical_cl_tra_eq2.loc[cl, tra]])
            this_np_eq2_tiled = np.tile(this_np_eq2, (len(cuff_res_df_agg.loc[cl]), 1))
            this_np_eq2_tiled[:, 3] = np.array(list(cuff_res_df_agg.loc[cl])).reshape(1, len(cuff_res_df_agg.loc[cl]))

            this_np_eq3 = np.array(
                [method, this_gene, -1, cl, 3, tra, phs_3, 0, 0, fp, tp, fn, prec, reca, fscore,
                 logical_cl_tra_eq3.loc[cl, tra]])
            this_np_eq3_tiled = np.tile(this_np_eq3, (len(cuff_res_df_agg.loc[cl]), 1))
            this_np_eq3_tiled[:, 3] = np.array(list(cuff_res_df_agg.loc[cl])).reshape(1, len(cuff_res_df_agg.loc[cl]))

            this_comb_np = np.concatenate((this_np_eq2_tiled, this_np_eq3_tiled), axis=0)

            this_comb_pandas = pd.DataFrame(data=this_comb_np,
                                            columns=['method', 'gene', 'sample', 'cluster', 'equation',
                                                     'tr_id', 'phs_score', 'quant',
                                                     'quant_norm', 'FP', 'TP', 'FN', 'precision', 'recall', 'F1Score',
                                                     'closest'], index=range(this_comb_np.shape[0]))

            gene_phs_results = gene_phs_results.append(this_comb_pandas, ignore_index=True)

            gene_phs_results = gene_phs_results.sort_values(['sample', 'cluster', 'tr_id', 'equation'], ascending=True)

            gene_phs_results = gene_phs_results.astype({'sample': int, 'equation': int, 'phs_score': float,
                                                        'quant': float, 'quant_norm': float, 'FP': int, 'TP': int,
                                                        'FN': int, 'precision': float, 'recall': float,
                                                        'F1Score': float, 'closest': int})

    file_name = file_pattern_name + '_per_sample.csv'
    gene_phs_results.to_csv(os.path.join(save_path, file_name))
    print(os.path.join(save_path, file_name), 'saved.')

    eq2_results = gene_phs_results[gene_phs_results['equation'] == 2].reset_index()
    eq2_results = eq2_results[eq2_results['closest'] == 1]
    eq3_results = gene_phs_results[gene_phs_results['equation'] == 3].reset_index()
    eq3_results = eq3_results[eq3_results['closest'] == 1]

    FP2 = np.mean(eq2_results['FP'])
    TP2 = np.mean(eq2_results['TP'])
    FN2 = np.mean(eq2_results['FN'])
    precision2 = np.mean(eq2_results['precision'])
    recall2 = np.mean(eq2_results['recall'])
    F1Score2 = np.mean(eq2_results['F1Score'])
    phs_score2 = np.mean(eq2_results['phs_score'])
    quant2 = np.mean(eq2_results['quant'])

    FP3 = np.mean(eq3_results['FP'])
    TP3 = np.mean(eq3_results['TP'])
    FN3 = np.mean(eq3_results['FN'])
    precision3 = np.mean(eq3_results['precision'])
    recall3 = np.mean(eq3_results['recall'])
    F1Score3 = np.mean(eq3_results['F1Score'])
    phs_score3 = np.mean(eq3_results['phs_score'])
    quant3 = np.mean(eq3_results['quant'])

    return FP2, TP2, FN2, precision2, recall2, F1Score2, phs_score2, quant2, FP3, TP3, FN3, precision3, recall3, \
           F1Score3, phs_score3, quant3


def evaluate_gene(gene_results_path, tr_ex_int_df, gene_name, save_path):

    file_pattern_name = gene_name + '_results'
    meth = 'BAMIE'
    n_row = 16
    meth_cov_df = pd.DataFrame(data=0, columns=['method', 'gene', 'equation', 'metric', 'value'],
                               index=range(n_row))

    FP2, TP2, FN2, precision2, recall2, F1Score2, phs_score2, quant2, FP3, TP3, FN3, precision3, recall3, F1Score3, \
    phs_score3, quant3 = compute_gene_metrics_bamie(tr_ex_int_df, gene_results_path, gene_name, save_path, file_pattern_name)

    meth_cov_df.iloc[0, :] = meth, gene_name, 2, 'FP', FP2
    meth_cov_df.iloc[1, :] = meth, gene_name, 2, 'TP', TP2
    meth_cov_df.iloc[2, :] = meth, gene_name, 2, 'FN', FN2
    meth_cov_df.iloc[3, :] = meth, gene_name, 2, 'precision', precision2
    meth_cov_df.iloc[4, :] = meth, gene_name, 2, 'recall', recall2
    meth_cov_df.iloc[5, :] = meth, gene_name, 2, 'F1Score', F1Score2
    meth_cov_df.iloc[6, :] = meth, gene_name, 2, 'phs_score', phs_score2
    meth_cov_df.iloc[7, :] = meth, gene_name, 2, 'quant', quant2

    meth_cov_df.iloc[8, :] = meth, gene_name, 3, 'FP', FP3
    meth_cov_df.iloc[9, :] = meth, gene_name, 3, 'TP', TP3
    meth_cov_df.iloc[10, :] = meth, gene_name, 3, 'FN', FN3
    meth_cov_df.iloc[11, :] = meth, gene_name, 3, 'precision', precision3
    meth_cov_df.iloc[12, :] = meth, gene_name, 3, 'recall', recall3
    meth_cov_df.iloc[13, :] = meth, gene_name, 3, 'F1Score', F1Score3
    meth_cov_df.iloc[14, :] = meth, gene_name, 3, 'phs_score', phs_score3
    meth_cov_df.iloc[15, :] = meth, gene_name, 3, 'quant', quant3

    meth_cov_df.to_csv(os.path.join(save_path, file_pattern_name + '_overall.csv'))
    print(os.path.join(save_path, file_pattern_name + '_overall.csv file saved.'))


class MODEL(object):

    def __init__(self, eta, alpha, epsilon, r, s):
        self.eta = eta
        self.alpha = alpha
        self.epsilon = epsilon
        self.r = r
        self.s = s
        self.beta = None
        self.b = None
        self.theta = None
        self.pi = None
        self.z = None

    def initialize_vars(self, gene, n_k):
        self.init_nodes = find_initial_nodes(gene.nodes_df, n_k)
        Z_matrix = np.zeros([gene.n_d, gene.n_v, n_k], dtype=int)
        for doc in range(0, gene.n_d):
            for v in range(0, gene.n_v):
                tempz = np.random.randint(0, n_k, size=gene.document[doc, v])
                for k in range(0, n_k):
                    Z_matrix[doc, v, k] = np.count_nonzero(tempz == k)

        self.z_init = Z_matrix

        # Theta: distribution of the samples over clusters
        Theta = np.zeros([gene.n_d, n_k])

        for i in range(gene.n_d):
            temp_dir = np.array([np.nan])
            while np.isnan(sum(temp_dir)):
                temp_dir = np.random.dirichlet(self.alpha * np.ones(n_k))
            Theta[i] = temp_dir

        # pi: distribution initialization
        pi = np.random.beta(self.r, self.s, size=n_k)

        # b: distribution initialization
        b = np.zeros([n_k, gene.n_v], dtype=int)
        for k in range(n_k):
            init = [int(node) for node in self.init_nodes[k]]
            b[k, init] = 1

        # Beta: distribution of the Clusters over intron excisions
        Beta = np.zeros([n_k, gene.n_v])
        for k in range(n_k):
            # Beta[k, :] = np.random.dirichlet(eta * np.ones(N_V))
            temp_dirb = np.array([np.nan])
            while np.isnan(sum(temp_dirb)):
                temp_dirb = np.random.dirichlet(self.eta * np.ones(gene.n_v))
            Beta[k, :] = temp_dirb

        Beta[Beta < self.epsilon] = self.epsilon

        self.z = Z_matrix
        self.beta = Beta
        self.theta = Theta
        self.pi = pi
        self.b = b
        self.converged = False

    def make_run_info(self, gene, n_k, burn_in, convergence_checkpoint_interval, n_iter):
        self.run_info = {}
        self.run_info['N_V'] = gene.n_v
        self.run_info['N_D'] = gene.n_d
        self.run_info['N_K'] = n_k
        self.run_info['N_W'] = gene.n_w
        self.run_info['gene_mvc_id'] = gene.mvc
        gene_mvc = [gene.id2w_dict[i] for i in gene.mvc]
        self.run_info['gene_mvc'] = gene_mvc
        self.run_info['r'] = self.r
        self.run_info['s'] = self.s
        self.run_info['alpha'] = self.alpha
        self.run_info['eta'] = self.eta
        self.run_info['epsilon'] = self.epsilon
        self.run_info['min_k'] = gene.min_k
        self.run_info['samples_df'] = gene.samples_df
        self.run_info['gene'] = gene.name
        self.run_info['gene_intersection'] = gene.intersection
        self.run_info['w2id_dict'] = gene.w2id_dict
        self.run_info['id2w_dict'] = gene.id2w_dict
        self.run_info['MIS'] = gene.mis
        self.run_info['max_ind_set'] = gene.max_ind_set
        self.run_info['init_nodes'] = self.init_nodes
        self.run_info['burn_in'] = burn_in
        self.run_info['convergence_checkpoint_interval'] = convergence_checkpoint_interval
        self.run_info['n_iter'] = n_iter
        self.run_info['convergence_point'] = n_iter
        self.run_info['document'] = gene.document
        self.run_info['document_tr'] = gene.document_tr
        self.run_info['document_te'] = gene.document_te
        self.run_info['tr_idx'] = gene.training_idx
        self.run_info['te_idx'] = gene.test_idx

    def is_converged_fwsr(self, likelihood, threshold=0.005):
        n0 = int(len(likelihood) / 2)
        this_ess = az.ess(np.array(likelihood[n0:]), method="quantile", prob=0.95)
        indices = range(n0, len(likelihood), int(this_ess))
        if len(indices) < 4:
            return False
        relevant_likelihood = [likelihood[i] for i in indices]
        sigma_hat_g_n = np.std(relevant_likelihood)
        honest_metric = sigma_hat_g_n / np.sqrt(len(indices)) + (1 / len(indices))
        mean_g_n = np.mean(relevant_likelihood)
        conv = honest_metric < np.abs(mean_g_n * threshold)
        return conv

    def log_likelihood(self):
        N_D = self.z.shape[0]
        N_V = self.z.shape[1]
        N_K = self.z.shape[2]

        beta_rel = self.beta > self.epsilon
        beta_rel = beta_rel.T
        bet = np.repeat(beta_rel[np.newaxis, :, :], N_D, axis=0)
        rel_dim = bet * (self.z > self.epsilon)
        Z_matrix_new = self.z.swapaxes(1, 2).reshape(N_D * N_K, N_V)

        rel_dim_new = rel_dim.swapaxes(1, 2).reshape(N_D * N_K, N_V)

        A = Z_matrix_new * rel_dim_new
        Z_cut = A[~np.all(A == 0, axis=1)]
        Beta_rep = np.repeat(self.beta[np.newaxis, :, :], N_D, axis=0).reshape(N_D * N_K, N_V)
        Beta_relevant = Beta_rep * rel_dim_new
        Beta_cut = Beta_relevant[~np.all(A == 0, axis=1)]
        Beta_cut = Beta_cut / (np.sum(Beta_cut, axis=1).reshape(-1, 1))  # normalize
        multinomial_pmf = gammaln(np.sum(Z_cut, axis=1) + 1) + np.sum(xlogy(Z_cut, Beta_cut) - gammaln(Z_cut + 1),
                                                                      axis=-1)

        likelihood = np.sum(multinomial_pmf)
        return likelihood

    def log_likelihood_te(self, document_te):
        N_K = self.run_info['N_K']
        likelihood_te = 0
        for i in range(document_te.shape[0]):
            for k in range(N_K):
                x = document_te[i, :]
                relevant_beta = list(np.where(self.beta[k, :] > self.epsilon)[0])
                relevant_lambda = list(np.where(x > self.epsilon)[0])
                relevant_dim = list(set(relevant_beta).intersection(set(relevant_lambda)))
                if len(relevant_dim) > 0:
                    temp_x = list(x[relevant_dim])
                    temp_beta = list(self.beta[k, relevant_dim])
                    likelihood_te += multinomial.logpmf(temp_x, np.sum(temp_x), temp_beta)
        likelihood_te = likelihood_te / (document_te.shape[0] * N_K)
        return likelihood_te

    def update_z(self):
        # Sample from full conditional of Z
        # save for computing relative error
        self.beta = adjust_matrices(self.beta, self.epsilon)
        self.theta = adjust_matrices(self.theta, self.epsilon)

        for doc in range(0, self.run_info['document'].shape[0]):
            for v in range(0, self.run_info['N_V']):

                # self.beta[:, v] = self.adjust_beta(v)

                # self.beta[:, v] = np.array([self.epsilon if self.beta[tem, v] < self.epsilon else self.beta[tem, v] \
                #                        for tem in range(self.beta[:, v].shape[0])])
                # self.theta[doc, :] = self.adjust_theta(doc)

                # self.theta[doc, :] = np.array([self.epsilon if self.theta[doc, tem] < self.epsilon else self.theta[doc, tem] \
                #                           for tem in range(self.theta[doc, :].shape[0])])

                ratio_v = np.exp(np.log(self.theta[doc, :]) + np.log(self.beta[:, v]))
                ratio_v /= np.sum(ratio_v)

                tempz = np.random.multinomial(1, ratio_v, size=self.run_info['document'][doc, v]).argmax(axis=1)

                for k in range(0, self.run_info['N_K']):
                    self.z[doc, v, k] = np.count_nonzero(tempz == k)

            # np.sum(self.z[doc,:,k]) counts the number of words in document doc which are assigned to cluster k

        # self.z = update_z_loop_numba(self.beta, self.theta, self.run_info['document_tr'].shape[0], self.run_info['N_V'],
        #                         self.run_info['N_K'], self.run_info['document_tr'])


    def update_theta(self):

        # Sample from full conditional of Theta
        for doc in range(self.run_info['N_D']):
            self.theta[doc, :] = np.random.dirichlet(self.alpha + np.sum(self.z[doc, :, :], axis=0))
        self.theta[self.theta < self.epsilon] = self.epsilon

    def update_pi(self):
        # update for pi
        m = np.sum(self.b, axis=1)
        for k in range(self.run_info['N_K']):
            self.pi[k] = np.random.beta(self.r + m[k], self.s + self.run_info['N_V'] - m[k], size=None)
            # pi[k] = np.random.beta(r + np.sum(Z_matrix[:, :, k]), s + np.sum(document) -
            # np.sum(Z_matrix[:, :, k]), size=None)

    def update_b(self):
        N_S = 10
        if not self.converged:
            for k in range(self.run_info['N_K']):
                random_clusters = sample_local_ind_set(self.run_info['gene_intersection'], self.run_info['N_V'], N_S,
                                                       self.b[k, :], self.beta[k, :], self.run_info['MIS'])

                unnorm_p_phi = np.zeros([len(random_clusters)])
                for t in range(len(random_clusters)):
                    cluster = random_clusters[t]
                    cluster_neighbor = list(np.where(np.sum(self.run_info['gene_intersection'][cluster, :] != 0, axis=0))[0])

                    term1 = sci_beta.logpdf(x=self.pi[k], a=self.r + len(cluster), b=self.s + len(cluster_neighbor), loc=0,
                                                    scale=1)
                    relevant_indices = list(set(range(self.run_info['N_V'])) - set(cluster_neighbor))
                    relevant_indices = np.sort(relevant_indices)
                    b_eta = self.eta * self.b[k, :]
                    b_eta_eps = np.array([v + self.epsilon if np.abs(v) < self.epsilon else v for v in list(b_eta)])
                    temp3 = np.array([v + self.epsilon if np.abs(v) < self.epsilon else v for v in list(self.beta[k, :])])
                    term2 = dirichlet.logpdf(temp3 / np.sum(temp3), b_eta_eps)
                    p_phi = np.exp(term1 + term2)
                    unnorm_p_phi[t] = np.nan_to_num(p_phi)

                norm_p_phi = np.nan_to_num(unnorm_p_phi / np.sum(unnorm_p_phi))

                pop_no = 10
                pop = []
                for po in range(pop_no):
                    sel_cluster = np.random.multinomial(1, norm_p_phi, size=1)[0]
                    new_cluster_idx = np.where(sel_cluster)[0][0]
                    pop.append(new_cluster_idx)
                new_cluster_idx = max(set(pop), key=pop.count)
                new_cluster = random_clusters[new_cluster_idx]
                temp = np.zeros([self.run_info['N_V']], dtype=int)
                temp[new_cluster] = 1
                self.b[k, :] = deepcopy(temp)

    def update_beta(self):

        # Sample from full conditional of Beta
        # Z_matrix[:, v, k] counts the number of times word v is assigned to cluster k throughout the whole corpus
        for k in range(self.run_info['N_K']):
            temp_b = np.array([v + self.epsilon if v == 0 else v for v in list(self.b[k, :])])
            self.beta[k, :] = np.random.dirichlet(temp_b * self.eta + np.sum(self.z[:, :, k], axis=0))

    def update_run_info(self, t, gg):
        self.run_info['gibbs'][t]['Theta'] = deepcopy(self.theta)
        self.run_info['gibbs'][t]['Beta'] = deepcopy(self.beta)
        self.run_info['gibbs'][t]['b'] = deepcopy(self.b)
        self.run_info['gibbs'][t]['error'] = np.sum(np.abs(self.z - self.z_init)) / (
                    self.run_info['N_D'] * self.run_info['N_W'])
        self.run_info['gibbs'][t]['likelihood_i'] = self.log_likelihood()
        self.run_info['gibbs'][t]['likelihood_te'] = self.log_likelihood_te(gg.document_te)

        if t == 0:
            self.run_info['gibbs'][t]['relative_error'] = np.sum(np.abs(self.z - self.z_init)) / (
                    self.run_info['N_D'] * self.run_info['N_W'])
        else:
            self.run_info['gibbs'][t]['relative_error'] = np.sum(
                np.abs(self.z - self.run_info['gibbs'][t - 1]['Z'])) / (self.run_info['N_D'] * self.run_info['N_W'])

        if self.converged or t >= burn_in:
            self.run_info['gibbs'][t]['Z'] = deepcopy(self.z)
        else:
            self.run_info['gibbs'][t]['Z'] = 0

    def get_log_likelihood_vec(self):
        runs_dict = self.run_info['gibbs']
        likelihood = []
        for i in runs_dict.keys():
            likelihood.append(runs_dict[i]['likelihood_i'])
        return likelihood

    def train(self, gene, n_k, n_iter, burn_in, convergence_checkpoint_interval, verbose):

        self.initialize_vars(gene, n_k)
        self.make_run_info(gene, n_k, burn_in, convergence_checkpoint_interval, n_iter)
        self.run_info['gibbs'] = {}

        startiter = time.time()
        it = 0
        while it <= min(self.run_info['convergence_point'] + 100, n_iter):

            self.run_info['gibbs'][it] = {}

            self.update_z()

            self.update_theta()

            self.update_pi()

            self.update_b()

            self.update_beta()

            self.update_run_info(it, gene)

            if it >= burn_in and it % convergence_checkpoint_interval == 0 and not self.converged:

                log_likelihood_vector = self.get_log_likelihood_vec()
                self.converged = self.is_converged_fwsr(log_likelihood_vector, threshold=0.005)

                if self.converged:
                    self.run_info['convergence_point'] = it

            if it % 100 == 0 and verbose:

                print('Gene', gene.name, ', Iteration', it, ', Likelihood =',
                      round(self.run_info['gibbs'][it]['likelihood_i'], 4), ', Converged:', self.converged)
            it += 1

        self.run_info['duration'] = round(time.time() - startiter, 3)
        self.run_info['duration_per_iter'] = round(self.run_info['duration'] / n_iter, 3)
        self.run_info['error'] = np.sum(np.abs(self.z - self.z_init)) / (self.run_info['N_D'] * self.run_info['N_W'])
        self.run_info['likelihood_te'] = self.log_likelihood_te(gene.document_te)


class GENE(object):

    def __init__(self, name, gene_list_dir):
        self.name = name
        self.junc_path = gene_list_dir + name + '/'
        self.result_path = gene_list_dir + 'results_' + self.name
        self.samples_df, self.samples_df_dict = self.get_sample_df()
        self.nodes_df = self.get_junctions()
        self.min_k = find_min_clusters(self.nodes_df)
        self.trainable = self.is_trainable()
        self.all_n_k = list(range(self.min_k, self.min_k + 19)) if self.trainable else []
        self.intersection = None
        self.overlap_m = None
        self.mvc = None
        self.word_dict = None
        self.document = None
        self.id2w_dict = None
        self.w2id_dict = None
        self.n_w_list = None
        self.n_w = None
        self.n_v = None
        self.n_d = None
        self.document_tr = None
        self.document_te = None
        self.training_idx = None
        self.test_idx = None
        self.mis = None
        self.max_ind_set = None
        # self.samples_df = None
        # self.samples_df_dict = None
        # self.nodes_df = None
        # self.min_k = None

    def get_sample_df(self):
        junc_files_list = listdir(self.junc_path)
        samples_list = [s for s in junc_files_list if self.name + '_' in s and '.junc' in s]
        samples_df = pd.DataFrame(dtype=int, columns=['chrom', 'chromEnd', 'chromStart',
                                                      'score', 'strand'])
        for sample in samples_list:
            if '.gz' in sample:
                with gzip.open(self.junc_path + sample) as f:
                    sample_df = pd.read_csv(f, sep='\t')
                    samples_df = samples_df.append(sample_df, ignore_index=True)
            else:
                sample_df = pd.read_csv(self.junc_path + sample, sep='\t')
                samples_df = samples_df.append(sample_df, ignore_index=True)
        if len(samples_df) == 0:
            return [], []
        else:

            samples_df = samples_df.groupby(['chromEnd', 'chromStart'])['score'].sum().reset_index()

            samples_df_dict = {}
            for i in range(len(samples_df)):
                samples_df_dict[i] = {}
                for ke in list(samples_df.columns):
                    samples_df_dict[i][ke] = samples_df.loc[i, ke]
            return samples_df, samples_df_dict

    def get_junctions(self):
        """Generate an interval graph,
        node_n = number of nodes in the generated graph
        Irange: (integer): The range, in which the intervals fall into"""

        junc_num = self.samples_df.shape[0]
        nodes_df = pd.DataFrame(data=np.zeros([junc_num, 3]), dtype=int,
                                columns=['start', 'length', 'end'],
                                index=range(0, junc_num))

        nodes_df['start'] = self.samples_df.chromStart
        nodes_df['end'] = self.samples_df.chromEnd
        nodes_df['length'] = nodes_df['end'] - nodes_df['start']

        nodes_df = nodes_df.sort_values(by=['end'])
        nodes_df = nodes_df.reset_index(drop=True)
        # nodes_df['label'] = nodes_df.index.values
        graph_labels = []
        node_labels = []

        for v in range(nodes_df.shape[0]):
            graph_labels.append(str(int(nodes_df.loc[v, 'start'])) + '_' + str(int(nodes_df.loc[v, 'end'])))
            node_labels.append(str(v))

        nodes_df['graph_labels'] = graph_labels
        nodes_df['node_labels'] = node_labels
        return nodes_df

    def get_conflict(self):
        """Find the intervals that have intersection"""
        intersection_m = np.zeros([self.nodes_df.shape[0], self.nodes_df.shape[0]], dtype=int)
        overlap_m = np.zeros([self.nodes_df.shape[0], self.nodes_df.shape[0]])
        for v1 in range(self.nodes_df.shape[0]):
            s1 = self.nodes_df.loc[v1, 'start']
            e1 = self.nodes_df.loc[v1, 'end']
            for v2 in range(v1 + 1, self.nodes_df.shape[0]):
                s2 = self.nodes_df.loc[v2, 'start']
                e2 = self.nodes_df.loc[v2, 'end']
                if e1 > s2 and s1 < e2:
                    intersection_m[v1, v2] = 1
                    intersection_m[v2, v1] = 1
                    overlap_percentage = (min([e1, e2]) - max([s1, s2])) / ((e2 - s2 + e1 - s1) / 2)
                    overlap_m[v1, v2] = overlap_percentage
                    overlap_m[v2, v1] = overlap_percentage

        return intersection_m, overlap_m

    def get_document(self):  # preprocess_gene_opt
        junc_files_list = listdir(self.junc_path)
        # gene_word_dict = {}
        gene_word_dict = {self.name: {}}
        samples_list = [s for s in junc_files_list if self.name + '_' in s and '.junc' in s]
        valid_samples = []
        for sample in samples_list:
            if '.gz' in sample:
                with gzip.open(self.junc_path + sample) as f:
                    sample_df = pd.read_csv(f, sep='\t')
            else:
                sample_df = pd.read_csv(self.junc_path + sample, sep='\t')

            if sample_df.shape[0] > 0:
                valid_samples.append(sample)
                gene_word_dict[self.name][sample] = {}
                sample_df = sample_df.groupby(['chromStart', 'chromEnd'])['score'].sum().reset_index()

                for idx, row in sample_df.iterrows():
                    start_row = row.chromStart
                    end_row = row.chromEnd
                    gene_word_dict[self.name][sample][str(start_row) + '-' + str(end_row)] = row.score

        w2id_dict = {}
        id2w_dict = {}

        for i in range(self.nodes_df.shape[0]):
            word = str(int(self.nodes_df.loc[i, 'start'])) + '-' + str(int(self.nodes_df.loc[i, 'end']))
            id2w_dict[i] = word
            w2id_dict[word] = i
        n_v = self.nodes_df.shape[0]
        document = np.zeros([len(valid_samples), n_v], dtype=int)
        for sample_id in range(len(valid_samples)):
            for key in gene_word_dict[self.name][valid_samples[sample_id]]:
                document[sample_id, w2id_dict[key]] = gene_word_dict[self.name][valid_samples[sample_id]][key]

        return gene_word_dict, document, id2w_dict, w2id_dict

    def is_trainable(self):
        # self.samples_df, self.samples_df_dict = self.get_sample_df()
        if len(self.samples_df) == 0:
            return False
        else:
            # self.nodes_df = self.get_junctions()
            # self.min_k = find_min_clusters(self.nodes_df)
            if self.min_k < 2:
                return False
            else:
                return True

    def preprocess(self):
        self.intersection, self.overlap_m = self.get_conflict()
        self.mvc = generalized_min_node_cover(self.intersection, i=2)
        self.word_dict, self.document, self.id2w_dict, self.w2id_dict = self.get_document()
        self.n_w_list = list(np.sum(self.document, axis=1))
        self.n_w = np.mean(self.n_w_list)
        self.n_v = self.nodes_df.shape[0]
        self.n_d = self.document.shape[0]
        self.document_tr, self.document_te, self.training_idx, self.test_idx = split_training_test(self.document,
                                                                                                   tr_percentage=95)
        self.mis, self.max_ind_set = find_mis(self.nodes_df)



if __name__ == '__main__':


    if '-k' in sys.argv:
        n_k = int(sys.argv[sys.argv.index('-k') + 1])
    else:
        n_k = 16

    if '-a' in sys.argv:
        main_path = sys.argv[sys.argv.index('-a') + 1]
    else:
        main_path = ''
        exit('Error: Select the main directory containing genes by -a.')

    if '-g' in sys.argv:
        gene_name = sys.argv[sys.argv.index('-g') + 1]
    else:
        gene_name = 'A2ML1'

    if '-r' in sys.argv:
        path_to_reference_file = sys.argv[sys.argv.index('-r') + 1]
    else:
        path_to_reference_file = ''
        exit('Error: Select the path to reference file by -r.')

    # Model parameters
    max_n_iter = 1000
    burn_in = 500
    convergence_checkpoint_interval = 50
    eta = 0.01
    alpha = 1
    epsilon = 0.000001
    r = 1
    s = 1

    # my_gene_tr_intron = 'D:\\UCONN\\Isoform\\Junctions\\final_gene_tr_intron_sim.csv'
    tr_ex_int_df = pd.read_csv(path_to_reference_file, index_col=0)

    with zipfile.ZipFile(os.path.join(main_path, gene_name)+'.zip', 'r') as zip_ref:
        zip_ref.extractall(main_path)

    print('training gene', gene_name, 'with k =', n_k)
    model = MODEL(eta=eta, alpha=alpha, epsilon=epsilon, r=r, s=s)

    gene = GENE(gene_name, main_path)

    gene.preprocess()

    model.train(gene, n_k, n_iter=max_n_iter, burn_in=burn_in,
                convergence_checkpoint_interval=convergence_checkpoint_interval, verbose=True)

    result_path = save_results(gene, model)

    evaluate_gene(result_path, tr_ex_int_df, gene_name, gene.result_path)
