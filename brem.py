import time, os, gzip, pickle, random, sys, zipfile
import numpy as np
import pandas as pd
import numba as nb
import arviz as az
import networkx as nx
from numba import jit
from os import listdir
from scipy.stats import beta as sci_beta
from scipy.stats import dirichlet, multinomial
from scipy.special import gammaln, xlogy
from copy import deepcopy
from collections import Counter


def computeDF(n_sample, effective_k,n_introns,result_df,gene_name,z_matrix,starts,ends):
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
    return G


def split_training_test(document_orig, tr_percentage=95):
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

    computeDF_vectorized(n_sample, effective_k, n_introns, result_df, gene_name, z_matrix, starts, ends)


    file_name_2 = 'brem_' + gene_name + '_K_' + str(effective_k) + '.csv'
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

                ratio_v = np.exp(np.log(self.theta[doc, :]) + np.log(self.beta[:, v]))
                ratio_v /= np.sum(ratio_v)

                tempz = np.random.multinomial(1, ratio_v, size=self.run_info['document'][doc, v]).argmax(axis=1)

                for k in range(0, self.run_info['N_K']):
                    self.z[doc, v, k] = np.count_nonzero(tempz == k)


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

    def update_run_info(self, t, gg, burn_in):
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

            self.update_run_info(it, gene, burn_in)

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


def main(n_k, max_n_iter, eta, alpha, r, s, main_path, gene_name):
    burn_in = max_n_iter/2
    convergence_checkpoint_interval = (max_n_iter - burn_in)/10
    epsilon = 0.000001

    # Read gene junction files
    with zipfile.ZipFile(os.path.join(main_path, gene_name)+'.zip', 'r') as zip_ref:
        zip_ref.extractall(main_path)

    # Make the model and gene objects
    print('training gene', gene_name, 'with k =', n_k)
    model = MODEL(eta=eta, alpha=alpha, epsilon=epsilon, r=r, s=s)

    gene = GENE(gene_name, main_path)

    # Preprocess the gene
    gene.preprocess()

    # Train the gene
    model.train(gene, n_k, n_iter=max_n_iter, burn_in=burn_in,
                convergence_checkpoint_interval=convergence_checkpoint_interval, verbose=True)

    # Save all the results, including all the parameters in the model in a pickle file and clusters
    _ = save_results(gene, model)


if __name__ == '__main__':

    if '-k' in sys.argv:
        n_k = int(sys.argv[sys.argv.index('-k') + 1])
    else:
        n_k = 3

    if '-i' in sys.argv:
        max_n_iter = int(sys.argv[sys.argv.index('-i') + 1])
    else:
        max_n_iter = 1000

    if '-eta' in sys.argv:
        eta = sys.argv[sys.argv.index('-eta') + 1]
    else:
        eta = 0.01

    if '-alpha' in sys.argv:
        alpha = sys.argv[sys.argv.index('-alpha') + 1]
    else:
        alpha = 1

    if '-r' in sys.argv:
        r = sys.argv[sys.argv.index('-r') + 1]
    else:
        r = 1

    if '-s' in sys.argv:
        s = sys.argv[sys.argv.index('-s') + 1]
    else:
        s = 1

    if '-a' in sys.argv:
        main_path = sys.argv[sys.argv.index('-a') + 1]
    else:
        main_path = ''

    if '-g' in sys.argv:
        gene_name = sys.argv[sys.argv.index('-g') + 1]
    else:
        gene_name = 'A2ML1'


    main(n_k, max_n_iter, eta, alpha, r, s, main_path, gene_name)
