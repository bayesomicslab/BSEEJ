import argparse
import sys

from BSEEJ.gene import Gene
from BSEEJ.model import Model
from utilities import *


class Main(object):
    """ Initializes the input values """
    n_cluster = 1
    max_n_iter = 1000
    eta = 0.01
    alpha = 1
    r = 1
    s = 1
    p = ''
    g = 'A2ML1'
    o = ''
    @classmethod
    def main(cls, cmd_args):
        """
        The main function sets the hyper-parameters values, accordingly initilizes BREM algorithm,
        then makes the model and saves the results.
        """
    
        cls.init(cmd_args)
    
        print('=====================================================')
        print('Gene:', cls.g)
        print('junction path:', cls.p)
    
        print('result path:', cls.o)
    
        print('Number of clusters:', cls.n_cluster)
        print('Maximum number of iterations:', cls.max_n_iter)
    
        print('model parameter, eta:', cls.eta)
        print('model parameter, alpha:', cls.alpha)
        print('model parameter, r:', cls.r)
        print('model parameter, s:', cls.s)
        print('=====================================================')
    
        burn_in = cls.max_n_iter / 2
        convergence_checkpoint_interval = (cls.max_n_iter - burn_in) / 10
        epsilon = 0.000001
    
        # Read gene junction files
        # with zipfile.ZipFile(os.path.join(cls.p, cls.g) + '.zip', 'r') as zip_ref:
        #     zip_ref.extractall(cls.p)
    
        # Make the model and gene objects
        print('training gene', cls.g, 'with k =', cls.n_cluster)
        model = Model(eta=cls.eta, alpha=cls.alpha, epsilon=epsilon, r=cls.r, s=cls.s)
    
        gene = Gene(cls.g, cls.p, cls.o)
    
        # Preprocess the gene
        gene.preprocess()
        
        # Train the gene
        model.train(gene, cls.n_cluster, n_iter=cls.max_n_iter, burn_in=burn_in,
                    convergence_checkpoint_interval=convergence_checkpoint_interval, verbose=True)
        
        # Save all the results, including all the parameters in the model in a pickle file and clusters
        _ = save_results(gene, model)
    
    @classmethod
    def init(cls, cmd_args):
        """ Check the parser for possible inputs and overrides the existing default values if any. """
        parser = Main.get_parser()
        args = parser.parse_args(cmd_args[1:])
        
        cls.n_cluster = int(args.n_cluster)
        cls.max_n_iter = int(args.max_n_iter)
        cls.eta = args.eta
        cls.alpha = args.alpha
        cls.r = args.r
        cls.s = args.s
        cls.p = args.main_path
        cls.g = args.gene_name
        cls.o = args.result_path
    
    @classmethod
    def get_parser(cls):
        parser = argparse.ArgumentParser(description='Implementation of BREM.')
        parser.add_argument("-k", "--n_cluster", help="Number of clusters (integer >= 1, default = 1)",
                            default=1)
        parser.add_argument("-i", "--max_n_iter", help="Max number of iterations (integer) (default = 1000)",
                            default=cls.max_n_iter)
        parser.add_argument("-e", "--eta", required=False, help="eta (default = 0.01)", default=0.01)
        parser.add_argument("-a", "--alpha", required=False, help="alpha (default = 1)", default=1)
        parser.add_argument("-r", "--r", required=False, help="model parameter r (default = 1)", default=1)
        parser.add_argument("-s", "--s", required=False, help="model parameter s (default = 1)", default=1)
        parser.add_argument("-p", "--main_path", required=False, help="Main path (default = A2ML1/)", default='A2ML1/')
        parser.add_argument("-o", "--result_path", required=False, help="result path (default = ./)", default='./results')
        parser.add_argument("-g", "--gene_name", required=False, help="gene_name (default = A2ML1)",
                            default='A2ML1')
        return parser



if __name__ == '__main__':
    Main.main(sys.argv)
