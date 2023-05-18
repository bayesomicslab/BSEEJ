import sys

if __name__ == '__main__':

    if '-k' in sys.argv:
        n_k = int(sys.argv[sys.argv.index('-k') + 1])
    else:
        n_k = 1

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
