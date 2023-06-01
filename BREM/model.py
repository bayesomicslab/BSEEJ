import time

from scipy.special import gammaln, xlogy
from scipy.stats import beta as sci_beta
from scipy.stats import dirichlet, multinomial

from utilities import *


class Model(object):
    
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
        self.init_nodes = None
        self.z = None
        self.beta = None
        self.theta = None
        self.pi = None
        self.b = None
        self.converged = None
        self.z_init = None
        self.run_info = None

    def initialize_vars(self, gene, n_k):
        """This function initializes model parameters and other variables for training (Gibbs)"""
        self.init_nodes = find_initial_nodes(gene.nodes_df, n_k)
        z_matrix = np.zeros([gene.n_d, gene.n_v, n_k], dtype=np.int32)
        for doc in range(0, gene.n_d):
            for v in range(0, gene.n_v):
                tempz = np.random.randint(0, n_k, size=gene.document[doc, v])
                for k in range(0, n_k):
                    z_matrix[doc, v, k] = np.count_nonzero(tempz == k)
        self.z_init = z_matrix
    
        # theta: distribution of the samples over clusters
        theta = np.zeros([gene.n_d, n_k])
    
        for i in range(gene.n_d):
            temp_dir = np.array([np.nan])
            while np.isnan(sum(temp_dir)):
                temp_dir = np.random.dirichlet(self.alpha * np.ones(n_k))
            theta[i] = temp_dir
    
        # pi: distribution initialization
        pi = np.random.beta(self.r, self.s, size=n_k)
    
        # b: distribution initialization
        b = np.zeros([n_k, gene.n_v], dtype=np.int32)
        for k in range(n_k):
            init = [int(node) for node in self.init_nodes[k]]
            b[k, init] = 1
    
        # beta: distribution of the Clusters over intron excisions
        beta = np.zeros([n_k, gene.n_v])
        for k in range(n_k):
            temp_dirb = np.array([np.nan])
            while np.isnan(sum(temp_dirb)):
                temp_dirb = np.random.dirichlet(self.eta * np.ones(gene.n_v))
            beta[k, :] = temp_dirb
    
        beta[beta < self.epsilon] = self.epsilon
    
        self.z = z_matrix
        self.beta = beta
        self.theta = theta
        self.pi = pi
        self.b = b
        self.converged = False
    
    def make_run_info(self, gene, n_k, burn_in, convergence_checkpoint_interval, n_iter):
        """this function saves all the information in the Gibbs runs"""
        self.run_info = dict()
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
    
    def log_likelihood(self):
        """Computes log likelihood at the end of each Gibbs iteration"""
        n_d = self.z.shape[0]
        n_v = self.z.shape[1]
        n_k = self.z.shape[2]
        
        beta_rel = self.beta > self.epsilon
        beta_rel = beta_rel.T
        bet = np.repeat(beta_rel[np.newaxis, :, :], n_d, axis=0)
        rel_dim = bet * (self.z > self.epsilon)
        z_matrix_new = self.z.swapaxes(1, 2).reshape(n_d * n_k, n_v)
        
        rel_dim_new = rel_dim.swapaxes(1, 2).reshape(n_d * n_k, n_v)
        
        aa = z_matrix_new * rel_dim_new
        z_cut = aa[~np.all(aa == 0, axis=1)]
        beta_rep = np.repeat(self.beta[np.newaxis, :, :], n_d, axis=0).reshape(n_d * n_k, n_v)
        beta_relevant = beta_rep * rel_dim_new
        beta_cut = beta_relevant[~np.all(aa == 0, axis=1)]
        beta_cut = beta_cut / (np.sum(beta_cut, axis=1).reshape(-1, 1))  # normalize
        multinomial_pmf = gammaln(np.sum(z_cut, axis=1) + 1) + np.sum(xlogy(z_cut, beta_cut) - gammaln(z_cut + 1),
                                                                      axis=-1)
        likelihood = np.sum(multinomial_pmf)
        return likelihood
    
    def log_likelihood_te(self, document_te):
        """Computes log likelihood of test"""
        n_k = self.run_info['N_K']
        likelihood_te = 0
        for i in range(document_te.shape[0]):
            for k in range(n_k):
                x = document_te[i, :]
                relevant_beta = list(np.where(self.beta[k, :] > self.epsilon)[0])
                relevant_lambda = list(np.where(x > self.epsilon)[0])
                relevant_dim = list(set(relevant_beta).intersection(set(relevant_lambda)))
                if len(relevant_dim) > 0:
                    temp_x = list(x[relevant_dim])
                    temp_beta = list(self.beta[k, relevant_dim])
                    likelihood_te += multinomial.logpmf(temp_x, np.sum(temp_x), temp_beta)
        likelihood_te = likelihood_te / (document_te.shape[0] * n_k)
        return likelihood_te
    
    def update_z(self):
        """Update z variable in the model"""
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
        """Update \theta variable in the model"""
    
        # Sample from full conditional of Theta
        for doc in range(self.run_info['N_D']):
            self.theta[doc, :] = np.random.dirichlet(self.alpha + np.sum(self.z[doc, :, :], axis=0))
        self.theta[self.theta < self.epsilon] = self.epsilon

    def update_pi(self):
        """Update \pi variable in the model"""
    
        # update for pi
        m = np.sum(self.b, axis=1)
        for k in range(self.run_info['N_K']):
            self.pi[k] = np.random.beta(self.r + m[k], self.s + self.run_info['N_V'] - m[k], size=None)
            # pi[k] = np.random.beta(r + np.sum(Z_matrix[:, :, k]), s + np.sum(document) -
            # np.sum(Z_matrix[:, :, k]), size=None)
    
    def update_b(self):
        """Update b variable in the model"""
        n_s = 10
        if not self.converged:
            for k in range(self.run_info['N_K']):
                random_clusters = sample_local_ind_set(self.run_info['gene_intersection'], self.run_info['N_V'], n_s,
                                                       self.b[k, :], self.beta[k, :], self.run_info['MIS'])
                
                unnorm_p_phi = np.zeros([len(random_clusters)])
                for t in range(len(random_clusters)):
                    cluster = random_clusters[t]
                    cluster_neighbor = list(
                        np.where(np.sum(self.run_info['gene_intersection'][cluster, :] != 0, axis=0))[0])
    
                    term1 = sci_beta.logpdf(x=self.pi[k], a=self.r + len(cluster), b=self.s + len(cluster_neighbor),
                                            loc=0,
                                            scale=1)
                    # relevant_indices = list(set(range(self.run_info['N_V'])) - set(cluster_neighbor))
                    # relevant_indices = np.sort(relevant_indices)
                    b_eta = self.eta * self.b[k, :]
                    b_eta_eps = np.array([v + self.epsilon if np.abs(v) < self.epsilon else v for v in list(b_eta)])
                    temp3 = np.array(
                        [v + self.epsilon if np.abs(v) < self.epsilon else v for v in list(self.beta[k, :])])
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
                temp = np.zeros([self.run_info['N_V']], dtype=np.int32)
                temp[new_cluster] = 1
                self.b[k, :] = deepcopy(temp)

    def update_beta(self):
        """Update \beta variable in the model"""
    
        # Sample from full conditional of Beta
        # Z_matrix[:, v, k] counts the number of times word v is assigned to cluster k throughout the whole corpus
        for k in range(self.run_info['N_K']):
            temp_b = np.array([v + self.epsilon if v == 0 else v for v in list(self.b[k, :])])
            self.beta[k, :] = np.random.dirichlet(temp_b * self.eta + np.sum(self.z[:, :, k], axis=0))

    def update_run_info(self, t, gg, burn_in):
        """saves Gibbs iteration info in the data"""
    
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
        """extract the values of likelihood from run info dictionary"""
    
        runs_dict = self.run_info['gibbs']
        likelihood = []
        for i in runs_dict.keys():
            likelihood.append(runs_dict[i]['likelihood_i'])
        return likelihood

    def train(self, gene, n_k, n_iter, burn_in, convergence_checkpoint_interval, verbose):
        """Run Gibbs sampling on the data"""
    
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
                self.converged = is_converged_fwsr(log_likelihood_vector, threshold=0.005)
                
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
