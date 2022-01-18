# Bayesian Reconstruction and Differential Testing of Intron Excision Sequences (BAMIE)
BAMIE code is provided in 'bamie.py' file.
The junction files for one gene (A2ML1) is added in the github.

To run BAMIE with default configurations, run the requirements.txt.
Then run 'bamie.py'.

Change the model parameters in 'bamie.py' if needed. 


## Usage
1. Clone the repository:
```sh
git clone https://github.com/aguiarlab/BAMIE.git
cd BAMIE
```

2. Install the project dependencies:
```sh
pip install -r requirements.txt
```

3. Run the code:
```sh
python3 bamie.py
```

## Model:
**BAMIE Probabilistic Graphical Model** <img src="./docs/model.png" width="500"> 


Main variables and parameters include:
* V is the set of unique intron excisions, indexed by v and its size of denoted by |V|.
* N is the number of samples and are indexed by i.
* J<sub>i</sub> is the number of intron excisions in i<sup>th</sup> sample. 
* K is the number of clusters (indexed by k).
* For the j<sup>th</sup> intron excision in the i<sup>th</sup> sample, we assign a cluster k. 
* Graph G = (V, E), where V is the set of unique intron excision and there is an edge between two intron excisions _iff_ they intersect each other.
* &Omega; is the set of all the independent sets in G.

* r, s are priors for &pi; Beta distribution:
![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Cpi_k%20%5Csim%20Beta%28r%2Cs%29%2C%20%5Cforall%20k%3D%5C%7B1%2C%20%5Cdots%2C%20K%5C%7D)
    * Increase in mean of Beta(r,s) results in increase in cluster size |SIE|.


* The structure of a (clusters) SIEs consists of the inclusion or exclusion of intron excisions.  
![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Cbegin%7Balign*%7D%20b_%7Bkv%7D%20%26%20%5Csim%20Bernoulli%28%5Cpi_k%29%2C%20%5Cforall%20v%5Cin%20C%20%5C%5C%20s.t.%20%26%20%5Chspace%7B20pt%7D%20%7Bb_%7Bk%5Ccdot%7D%7D%20%5Cin%20%5COmega%20%5Cend%7Balign*%7D)


* For cluster k, &\beta;<sub>k</sub> is a |V|-dimensional Dirichlet which represents the distribution of the cluster k over the intron excisions.
![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Cbeta_k%20%5Csim%20Dirichlet_%7B%7CV%7C%7D%28%7B%5Ceta%7D%20%5Codot%20%7Bb_%7Bk%7D%7D%29)
    * &eta; = (&eta;<sub>1</sub>, &eta;<sub>2</sub>, ..., &eta;<sub>|V|</sub>) is &beta; variable prior.

* For the ith sample, variable &theta;<sub>i</sub> is a K-dimensional Dirichlet distribution and represents the proportions of the clusters in sample i.
![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Ctheta_i%20%5Csim%20Dirichlet_K%28%7B%5Calpha%7D%29)
    * &alpha; = (&alpha;<sub>1</sub>, &alpha;<sub>2</sub>, ..., &alpha;<sub>N</sub>) is &theta; variable prior.

* Variable z<sub>{ij}</sub>$ is the cluster assignment for jth intron excision in ith sample. It can take a natural value between 1 and K and follows a Multinomial distribution:
![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Cbegin%7Balign*%7D%20z_%7Bij%7D%20%26%20%5Csim%20Multinomial%28%5Ctheta_i%29%20%5C%5C%20%26%20%7BZ%7D%20%5Cin%20%5C%7B1%2C%20%5Cdots%2C%20K%5C%7D%5E%7BN%20%5Ctimes%20J_i%7D%20%5Cend%7Balign*%7D)

* In the ith sample, the jth intron excision is w<sub>{ij}</sub> and is observed and follows a Multinomial distribution:
![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20w_%7Bij%7D%20%5Csim%20Multinomial%28%5Cbeta_%7Bz_%7Bij%7D%7D%29)
