import argparse
import sys
import zipfile

from BREM.gene import Gene
from BREM.model import Model
from utilities import *


class Main(object):
    
    @classmethod
    def main(self, cmd_args):
        """
        The main function sets the hyper-parameters values, accordingly initilizes BREM algorithm,
        then makes the model and saves the results.
        """
        
        self.init(cmd_args)
        
        print('=====================================================')
        print('Gene:', self.g)
        print('path: ./', self.p)
        
        print('Number of clusters:', self.n_cluster)
        print('Maximum number of iterations:', self.max_n_iter)
        
        print('model parameter, eta:', self.eta)
        print('model parameter, alpha:', self.alpha)
        print('model parameter, r:', self.r)
        print('model parameter, s:', self.s)
        print('=====================================================')
        
        burn_in = self.max_n_iter / 2
        convergence_checkpoint_interval = (self.max_n_iter - burn_in) / 10
        epsilon = 0.000001
        
        # Read gene junction files
        with zipfile.ZipFile(os.path.join(self.p, self.g) + '.zip', 'r') as zip_ref:
            zip_ref.extractall(self.p)
        
        # Make the model and gene objects
        print('training gene', self.g, 'with k =', self.n_cluster)
        model = Model(eta=self.eta, alpha=self.alpha, epsilon=epsilon, r=self.r, s=self.s)
        
        gene = Gene(self.g, self.p)
        
        # Preprocess the gene
        gene.preprocess()
        
        # Train the gene
        model.train(gene, self.n_cluster, n_iter=self.max_n_iter, burn_in=burn_in,
                    convergence_checkpoint_interval=convergence_checkpoint_interval, verbose=True)
        
        # Save all the results, including all the parameters in the model in a pickle file and clusters
        _ = save_results(gene, model)
    
    @classmethod
    def init(self, cmd_args):
        """ Initializes the input values """
        
        self.n_cluster = 1
        self.max_n_iter = 1000
        self.eta = 0.01
        self.alpha = 1
        self.r = 1
        self.s = 1
        self.main_path = ''
        self.gene_name = 'A2ML1'
        
        """ Check the parser for possible inputs and overrides the existing default values if any. """
        parser = Main.getParser()
        args = parser.parse_args(cmd_args[1:])
        
        self.n_cluster = int(args.n_cluster)
        self.max_n_iter = int(args.max_n_iter)
        self.eta = args.eta
        self.alpha = args.alpha
        self.r = args.r
        self.s = args.s
        self.p = args.main_path
        self.g = args.gene_name
    
    @classmethod
    def getParser(cls):
        parser = argparse.ArgumentParser(description='Implementation of BREM.')
        parser.add_argument("-k", "--n_cluster", help="Number of clusters (integer >= 1, default = 1)",
                            default=1)
        parser.add_argument("-i", "--max_n_iter", help="Max number of iterations (integer) (default = 1000)",
                            default=cls.max_n_iter)
        parser.add_argument("-e", "--eta", required=False, help="eta (default = 0.01)", default=cls.eta)
        parser.add_argument("-a", "--alpha", required=False, help="alpha (default = 1)", default=cls.alpha)
        parser.add_argument("-r", "--r", required=False, help="model parameter r (default = 1)", default=cls.r)
        parser.add_argument("-s", "--s", required=False, help="model parameter s (default = 1)", default=cls.s)
        parser.add_argument("-p", "--main_path", required=False, help="Main path (default = ./)", default=cls.main_path)
        parser.add_argument("-g", "--gene_name", required=False, help="gene_name (default = A2ML1)",
                            default=cls.gene_name)
    
        return parser


if __name__ == '__main__':
    Main.main(sys.argv)
