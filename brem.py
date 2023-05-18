import sys

from utilities import *


def main(n_k, max_n_iteration, eta_var, alpha_var, r_var, s_var, main_folder, gene_label):
    burn_in = max_n_iteration / 2
    convergence_checkpoint_interval = (max_n_iteration - burn_in) / 10
    epsilon = 0.000001
    
    # Read gene junction files
    with zipfile.ZipFile(os.path.join(main_folder, gene_label) + '.zip', 'r') as zip_ref:
        zip_ref.extractall(main_folder)
    
    # Make the model and gene objects
    print('training gene', gene_label, 'with k =', n_k)
    model = Model(eta=eta_var, alpha=alpha_var, epsilon=epsilon, r=r_var, s=s_var)
    
    gene = Gene(gene_label, main_folder)
    
    # Preprocess the gene
    gene.preprocess()
    
    # Train the gene
    model.train(gene, n_k, n_iter=max_n_iteration, burn_in=burn_in,
                convergence_checkpoint_interval=convergence_checkpoint_interval, verbose=True)
    
    # Save all the results, including all the parameters in the model in a pickle file and clusters
    _ = save_results(gene, model)


if __name__ == '__main__':
    
    if '-k' in sys.argv:
        n_cluster = int(sys.argv[sys.argv.index('-k') + 1])
    else:
        n_cluster = 1
    
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
    
    main(n_cluster, max_n_iter, eta, alpha, r, s, main_path, gene_name)
