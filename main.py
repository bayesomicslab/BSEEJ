




def main():
    # Parse command-line arguments 
    parser = argparse.ArgumentParser(description='Implementation of BSEEJ.')
    parser.add_argument("-k", "--n_cluster", help="Number of clusters (integer >= 1, default = 1)",
                        default=1)
    parser.add_argument("-i", "--max_n_iter", help="Max number of iterations (integer) (default = 1000)",
                        default=cls.max_n_iter)
    parser.add_argument("-e", "--eta", required=False, help="eta (default = 0.01)", default=cls.eta)
    parser.add_argument("-a", "--alpha", required=False, help="alpha (default = 1)", default=cls.alpha)
    parser.add_argument("-r", "--r", required=False, help="model parameter r (default = 1)", default=cls.r)
    parser.add_argument("-s", "--s", required=False, help="model parameter s (default = 1)", default=cls.s)
    parser.add_argument("-p", "--main_path", required=False, help="Main path (default = ./)", default=cls.p)
    parser.add_argument("-o", "--result_path", required=False, help="result path (default = ./)", default=cls.o)
    parser.add_argument("-g", "--gene_name", required=False, help="gene_name (default = A2ML1)",
                        default=cls.g)
    # parser.add_argument("--epsilon", help="epsilon in computing prob.")
    args = parser.parse_args()

    class Args:
        def __init__(self):
            self.n_cluster = 1
            self.max_n_iter = 1000
            self.eta = 0.01
            self.alpha = 1
            self.r = 1
            self.s = 1
            self.main_path = '/labs/Aguiar/BSEEJ/A2ML1'
            self.gene_name = 'A2ML1'
            self.result_path = '/labs/Aguiar/BSEEJ/results'
            # self.alleles = [0, 1]

    # Create the mock args object
    args = Args()

    n_cluster = int(args.n_cluster)
    max_n_iter = int(args.max_n_iter)
    eta = args.eta
    alpha = args.alpha
    r = args.r
    s = args.s
    p = args.main_path
    g = args.gene_name
    o = args.result_path


    print('=====================================================')
    print('Gene:', g)
    print('junction path:', p)

    print('result path:', o)

    print('Number of clusters:', n_cluster)
    print('Maximum number of iterations:', max_n_iter)

    print('model parameter, eta:', eta)
    print('model parameter, alpha:', alpha)
    print('model parameter, r:', r)
    print('model parameter, s:', s)
    print('=====================================================')

    burn_in = max_n_iter / 2
    convergence_checkpoint_interval = (max_n_iter - burn_in) / 10
    epsilon = 0.000001

    # Read gene junction files
    # with zipfile.ZipFile(os.path.join(cls.p, cls.g) + '.zip', 'r') as zip_ref:
    #     zip_ref.extractall(cls.p)

    # Make the model and gene objects
    print('training gene', g, 'with k =', n_cluster)
    model = Model(eta=eta, alpha=alpha, epsilon=epsilon, r=r, s=s)

    gene = Gene(g, p, o)

    # Preprocess the gene
    gene.preprocess()
    
    # Train the gene
    model.train(gene, n_cluster, n_iter=max_n_iter, burn_in=burn_in,
                convergence_checkpoint_interval=convergence_checkpoint_interval, verbose=True)
    
    # Save all the results, including all the parameters in the model in a pickle file and clusters
    _ = save_results(gene, model)
