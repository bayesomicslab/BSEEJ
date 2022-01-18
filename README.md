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

## Model Parameters
<img src="./docs/model.png" width="500"> **BAMIE Probabilistic Graphical Model**

Model Priors:


Main variables include:
* &alpha; is 
* &eta;
* r, s are priors for &pi; Beta distribution:

![](https://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Clarge%20%5Cpi_k%20%5Csim%20Beta%28r%2Cs%29%2C%20%5Cforall%20k%3D%5C%7B1%2C%20%5Cdots%2C%20K%5C%7D)

* V is the set of unique intron excisions, indexed by v and its size of denoted by |V|.
* N is the number of samples and are indexed by i.
* J<sub>i</sub> is the number of intron excisions in i<sup>th</sup> sample. 
* K is the number of clusters (indexed by k).
* For the j<sup>th</sup> intron excision in the i<sup>th</sup> sample, we assign a cluster k. 
* Graph G = (V, E), where V is the set of unique intron excision and there is an edge between two intron excisions _iff_ they intersect each other.
* &Omega; is the set of all the independent sets in G.
