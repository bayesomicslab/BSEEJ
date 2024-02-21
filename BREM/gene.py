from utilities import *


class Gene(object):
    
    def __init__(self, name, gene_list_dir):
        """Initialize gene instance from the zip file containing the gene .bam files:
        This function computes the gene nodes and the interval graph and minimum number of clusters"""
    
        self.name = name # Corresponds to the gene name
        self.junc_path = gene_list_dir + name + '/' # Corresponds to the path to the .junc file
        self.result_path = gene_list_dir + 'results_' + self.name # Sets a path to save model training results
        self.samples_df, self.samples_df_dict = self.get_sample_df() # Gets the dictionary and the dataframe built from the .junc files, remember columns are 'chromStart' 'chromEnd' 'score'
        self.nodes_df = self.get_junctions() #returns a modified dataframe with columns 'start' 'length' 'end' 'graph_labels' 'node_labels'
        self.min_k = find_min_clusters(self.nodes_df) # gets the clique number of the interval graph that as created
        self.trainable = self.is_trainable() # Determines if the model can be trained if the cluster size is above 1 meaning it's not the trivial case
        self.all_n_k = list(range(self.min_k, self.min_k + 19)) if self.trainable else [] # Sets a list indicating the number of components to search over (TODO: Still not completely clear about this) (TODO: Turn that + 19 into a parameter given by the user)

        # Lots of hyperparameters 
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

    # Gets the samples from the .junc files and saves them into a single datafram
    def get_sample_df(self):
        """computes the gene's intro excisions from .bam files."""
    
        junc_files_list = os.listdir(self.junc_path) # List all of the junction files in the path
        samples_list = [s for s in junc_files_list if self.name + '_' in s and '.junc' in s] # grab all of the junction files that correspond to the gene from the folder

        #Grabs all the samples, reads the individual csv files, and adds them to a single list called samples 
        samples = []
        for sample in samples_list:
            if '.gz' in sample:
                with gzip.open(self.junc_path + sample) as f:
                    sample = pd.read_csv(f, sep='\t').values.tolist()
                    samples.extend(sample)
            else:
                sample = pd.read_csv(self.junc_path + sample, sep='\t').values.tolist()
                samples.extend(sample)

        
        # Convert the lists of values into a dataframe, with columns chromStart chromEnd qual score strand
        # qual is dropped later
        # changes the chromStart to ints and chromEnd to ints and score to int
        samples_df = pd.DataFrame(samples, columns=['chrom', 'chromStart', 'chromEnd',
                                                      'qual', 'score', 'strand'])
        samples_df = samples_df[['chrom', 'chromStart', 'chromEnd','score', 'strand']]
        samples_df = samples_df.astype({"chromStart": np.int32, "chromEnd": np.int32, "score": np.int32})

        # If there aren't any strands then welp idk return nothing?
        if len(samples_df) == 0:
            return [], []
        else:
            # Groups all junctions with the same start and end into only one and adds up their individual scores. This guarantees there are no repeated junctions
            # This also drops 'chrom' and 'strand'
            samples_df = samples_df.groupby(['chromEnd', 'chromStart'])['score'].sum().reset_index()
            print(samples_df)
            
            # Builds a dictionary of dictionaries where the first key is the row and the second key is the column:
            #{
            #   1: {
            #       "chromEnd": 10101
            #       "chromStart": 12101
            #       "score": 2
            #   }
            #   ...
            #}
            samples_df_dict = {}
            for i in range(len(samples_df)):
                samples_df_dict[i] = {}
                for ke in list(samples_df.columns):
                    samples_df_dict[i][ke] = samples_df.loc[i, ke]

            # Return both the dataframe and the dictionary
            return samples_df, samples_df_dict
    
    # Creates a new dataframe with columns 'start' 'length' 'end' 'graph_labels' 'node_labels' where start is chromStart end is chromEnd length is the calculated length of the junction graph labels is a string of format f"{start}_{end}" and node_labels is the row number
    def get_junctions(self):
        """Generate an interval graph,
        node_n = number of nodes in the generated graph
        Irange: (integer): The range, in which the intervals fall into"""
        
        junc_num = self.samples_df.shape[0] # Gets the total number of unique junctions extracted from the files
        nodes_df = pd.DataFrame(data=np.zeros([junc_num, 3]), dtype=np.int32,
                                columns=['start', 'length', 'end'],
                                index=range(0, junc_num)) # Creates a dataframe where each row is a junction and the columns are 'start' 'length' 'end'
        
        nodes_df['start'] = self.samples_df.chromStart # Adds the chromStart from the past dataframe to the start of this new datafram
        nodes_df['end'] = self.samples_df.chromEnd # Adds the chromEnd from the past dataframe to the end of this new dataframe
        nodes_df['length'] = nodes_df['end'] - nodes_df['start'] # Calculate the length of the junction
        
        nodes_df = nodes_df.sort_values(by=['end']) # Sorts the values by their endings
        nodes_df = nodes_df.reset_index(drop=True) #fixed the dataframes index to reflect the sorting
        # nodes_df['label'] = nodes_df.index.values
        graph_labels = []
        node_labels = []
        
        # Iterate over all of the dataframe and make graph labels equal to the string f"{start}_{end}" where start and end are the values for the respective columns for tht row
        # iterates over the dataframe and adds as node_labels the number of the row
        for v in range(nodes_df.shape[0]):
            graph_labels.append(str(int(nodes_df.loc[v, 'start'])) + '_' + str(int(nodes_df.loc[v, 'end'])))
            node_labels.append(str(v))
        
        # Adds the two arraysjust created to the dataframe nodes
        nodes_df['graph_labels'] = graph_labels
        nodes_df['node_labels'] = node_labels

        # Returns the dataframe nodes
        return nodes_df
    

    # Calculate the overlap between all junctions and return adjacency matrices representing the intersect graph with their corresponding overlaps
    def get_conflict(self):
        """Find the intervals that have intersection"""
        intersection_m = np.zeros([self.nodes_df.shape[0], self.nodes_df.shape[0]], dtype=np.int32) # build an empty adjacency matrix of size num_junc x num_junc
        overlap_m = np.zeros([self.nodes_df.shape[0], self.nodes_df.shape[0]]) # build an empty adjacency matrix of size num_junc x num_junc

        
        for v1 in range(self.nodes_df.shape[0]): # Iterate over all the nodes
            s1 = self.nodes_df.loc[v1, 'start'] # assign the start of the junction to s1 
            e1 = self.nodes_df.loc[v1, 'end'] # assign the end of the junction to e1

            for v2 in range(v1 + 1, self.nodes_df.shape[0]): # Iterate over the other nodes
                s2 = self.nodes_df.loc[v2, 'start'] # Assign the start of the second junction to s1
                e2 = self.nodes_df.loc[v2, 'end'] # Assign the end of the second junction to e1

                if e1 > s2 and s1 < e2: # if the first junction intersects with the second junction (TODO: This should be >= and <= because of edge cases)
                    intersection_m[v1, v2] = 1 # build and edge between the two junction nodes
                    intersection_m[v2, v1] = 1 # build the edge between the two junction nodes
                    overlap_percentage = (min([e1, e2]) - max([s1, s2])) / ((e2 - s2 + e1 - s1) / 2) # calculate how much they overlap
                    overlap_m[v1, v2] = overlap_percentage # Save the amount of overlap in the overlap adjacency matrix
                    overlap_m[v2, v1] = overlap_percentage # Save the amount of overlap in the overlap adjacency matrix
        
        # Return the intersections and overlap percentages as adjacency matrices 
        return intersection_m, overlap_m
    
    def get_document(self):  # preprocess_gene_opt
        """Extract all samples information from non empty .bam files"""
        junc_files_list = os.listdir(self.junc_path) # load the dir where the junc files are
        gene_word_dict = {self.name: {}} # Build empty dictionary with the gene name as key
        samples_list = [s for s in junc_files_list if self.name + '_' in s and '.junc' in s] #determines the files to use as samples
        valid_samples = []
        for sample in samples_list:
            #Unzip if the filez are zipped
            if '.gz' in sample:
                with gzip.open(self.junc_path + sample) as f:
                    sample_df = pd.read_csv(f, sep='\t') # read the junc files as csv
            else:
                sample_df = pd.read_csv(self.junc_path + sample, sep='\t') # read the junc files as csv
            
            if sample_df.shape[0] > 0: # if there are rows to process
                valid_samples.append(sample) # Add the sample into the valid samples
                gene_word_dict[self.name][sample] = {} # add the sample into the dictionary
                sample_df = sample_df.groupby(['chromStart', 'chromEnd'])['score'].sum().reset_index() # Add up the scores of those junctions that start and end at the same place
                
                for idx, row in sample_df.iterrows(): #iterate over the samples
                    start_row = row.chromStart
                    end_row = row.chromEnd
                    gene_word_dict[self.name][sample][str(start_row) + '-' + str(end_row)] = row.score #build the dictionary such that per sample the junctions are saved into the dictionary ie
                    # {"gene_name": {"sample_filename": "junc_start-junc_end": score}}
        
        w2id_dict = {} # maps the junc_start-junc_end to the row in nodes_df that node is in
        id2w_dict = {} # does the opposite of the above
        
        for i in range(self.nodes_df.shape[0]): # Iterate over the adjacency list of the interval graph
            word = str(int(self.nodes_df.loc[i, 'start'])) + '-' + str(int(self.nodes_df.loc[i, 'end'])) # word is equal to the junc_start-junc_end
            id2w_dict[i] = word
            w2id_dict[word] = i
        n_v = self.nodes_df.shape[0] # Number of vertices
        document = np.zeros([len(valid_samples), n_v], dtype=np.int32) # build a matrix of the number of samples by the number of nodes
        for sample_id in range(len(valid_samples)): #Iterate over the samples
            for key in gene_word_dict[self.name][valid_samples[sample_id]]: # iterate over the junctions per sample
                document[sample_id, w2id_dict[key]] = gene_word_dict[self.name][valid_samples[sample_id]][key] # The document at the position of the sample_id and the id of the junction in the nodes_df (the number of the node) is equal to the score of the junction that that specific node in nodes_df has
        
        return gene_word_dict, document, id2w_dict, w2id_dict # return everything
    
    #d= Determines if the system is trainable because determines if the cluster size is less than 2 then every node would have it's own hyperparameter
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
        self.intersection, self.overlap_m = self.get_conflict() # Get adjacency matrices representing the intersection graph of the jucntions
        self.mvc = generalized_min_node_cover(self.intersection, i=2) # gets the minimum node cover for the interval graph that was created
        self.word_dict, self.document, self.id2w_dict, self.w2id_dict = self.get_document() # Returns a lot of things:
        # word_dict: is a dictionary where the first key is the name of the gene, the second key is the name of the file where a sample was taken from, the third key is the junc_start-junc_end of a specific junction and the value is the score of that specific junction
        # document is a matrix of the number of samples by the number of nodes in nodes_df (the interval graph) where the value in each cell is the score of the node in that specific sample

        self.n_w_list = list(np.sum(self.document, axis=1)) # Add the scores of every node per sample essentialy ending up with a list where each position is the ith node and it's score
        self.n_w = np.mean(self.n_w_list) # calculate the mean score
        self.n_v = self.nodes_df.shape[0] # number of vertices
        self.n_d = self.document.shape[0] # number of samples
        self.document_tr, self.document_te, self.training_idx, self.test_idx = split_training_test(self.document, tr_percentage=95) #splits the datasets into training and testing

        self.mis, self.max_ind_set = find_mis(self.nodes_df) # get the number of nodes in the max independent set and the max indepent set of the interval graph
