from utilities import *


class Gene(object):
    
    def __init__(self, name, gene_list_dir, result_path):
        """Initialize gene instance from the zip file containing the gene .bam files:
        This function computes the gene nodes and the interval graph and minimum number of clusters"""
    
        self.name = name
        # self.junc_path = gene_list_dir + name + '/'
        self.junc_path = gene_list_dir
        # self.result_path = gene_list_dir + 'results_' + self.name
        if not os.path.exists(result_path):
            os.mkdir(result_path)
        if not os.path.exists(os.path.join(result_path, self.name)):
            os.mkdir(os.path.join(result_path, self.name))
        self.result_path = os.path.join(result_path, self.name)
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
        """computes the gene's intro excisions from .bam files."""
        min_coverage = 30
        # junc_files_list = os.listdir(self.junc_path)
        # samples_list = [s for s in junc_files_list if self.name + '_' in s and '.junc' in s]
        samples_list = [os.path.join(self.junc_path, s) for s in os.listdir(self.junc_path) if '.junc' in s]
        columns = ['chrom', 'chromStart', 'chromEnd', 'junc_id', 'score', 'strand', 'start', 'end', 'f1', 'f2', 'f3',
                   'f4']
        samples_dfs = [pd.read_csv(file, sep='\t', names=columns) for file in samples_list]
        samples_df = pd.concat(samples_dfs)
        samples_df = samples_df[samples_df['score'] >= min_coverage].reset_index(drop=True)
        # samples = []
        # for sample in samples_list:
        #     if '.gz' in sample:
        #         with gzip.open(self.junc_path + sample) as f:
        #             sample = pd.read_csv(f, sep='\t').values.tolist()
        #             samples.extend(sample)
        #     else:
        #         sample = pd.read_csv(self.junc_path + sample, sep='\t').values.tolist()
        #         samples.extend(sample)
        #
        # samples_df = pd.DataFrame(samples, columns=['chrom', 'chromStart', 'chromEnd',
        #                                               'qual', 'score', 'strand'])
        samples_df = samples_df[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]
        samples_df = samples_df.astype({"chromStart": np.int32, "chromEnd": np.int32, "score": np.int32})
    
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
        nodes_df = pd.DataFrame(data=np.zeros([junc_num, 3]), dtype=np.int32,
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
        intersection_m = np.zeros([self.nodes_df.shape[0], self.nodes_df.shape[0]], dtype=np.int32)
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
        """Extract all samples information from non empty .bam files"""
    
        min_coverage = 30
        # junc_files_list = os.listdir(self.junc_path)
        # samples_list = [s for s in junc_files_list if self.name + '_' in s and '.junc' in s]
        # samples_list = [os.path.join(self.junc_path, s) for s in os.listdir(self.junc_path) if '.junc' in s]
        columns = ['chrom', 'chromStart', 'chromEnd', 'junc_id', 'score', 'strand', 'start', 'end', 'f1', 'f2', 'f3',
                   'f4']
        # samples_dfs = [pd.read_csv(file, sep='\t', names=columns) for file in samples_list]
        # samples_df = pd.concat(samples_dfs)
        # samples_df = samples_df[samples_df['score'] >= min_coverage].reset_index(drop=True)
        # junc_files_list = os.listdir(self.junc_path)
        gene_word_dict = {self.name: {}}
        samples_list = [os.path.join(self.junc_path, s) for s in os.listdir(self.junc_path) if '.junc' in s]
        valid_samples = []
        for sample in samples_list:
            if '.gz' in sample:
                # with gzip.open(sample) as f:
                sample_df = pd.read_csv(sample, names=columns, sep='\t', compression='gzip')
                sample_df = sample_df[sample_df['score'] >= min_coverage].reset_index(drop=True)
                sample_df = sample_df[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]
            else:
                sample_df = pd.read_csv(sample, names=columns, sep='\t')
                sample_df = sample_df[sample_df['score'] >= min_coverage].reset_index(drop=True)
                sample_df = sample_df[['chrom', 'chromStart', 'chromEnd', 'score', 'strand']]
        
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
        document = np.zeros([len(valid_samples), n_v], dtype=np.int32)
        for sample_id in range(len(valid_samples)):
            for key in gene_word_dict[self.name][valid_samples[sample_id]]:
                document[sample_id, w2id_dict[key]] = gene_word_dict[self.name][valid_samples[sample_id]][key]
        
        return gene_word_dict, document, id2w_dict, w2id_dict
    
    def is_trainable(self):
        """This function determines if the minimum number of clusters for a gene is less than 2 (trivial case)"""
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
        """This function computes initial properties of the interval graph before Gibbs runs."""
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
