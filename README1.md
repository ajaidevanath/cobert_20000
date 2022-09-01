## Architectures

The following repo gives 2 architectures for information retrieval on covid systems whcih can be merged as per requirement.

The first one is the BioSentVec model ( biosentvec.py) & the Information_Retrieval_COVID_BiosentVec[BioSentVec](https://github.com/ncbi-nlp/BioSentVec) which uses the clustering method on article titles along with kmeans and kNearest neighbours to get the closest article.
The closest article is fed into the customized BERT model(covidbert_last_layer_training) , which has been trained on by unfreezing the last 80- layers of the biobert model[bio-bert](https://arxiv.org/abs/1901.08746),(https://github.com/dmis-lab/biobert)
We then add the QA head on top of this model to train step by step on the [Squad-dataset](https://rajpurkar.github.io/SQuAD-explorer/) which contains annotated question answer pairs with context.

The SQUAD Dataset consists of 87000 different question answer pairs. Training them on a single epoch takes more than 72 hours. Hence we used step training method to train them at steps of 10,000 with a batch size of 16 and learnign rate of lre-5. The training time for each step was 8 hours on a 8GB GPU machine.

## Setup the BioSentVec model 

BioSentVec uses sent2vec model, hence we have to manually install the [Sent2vecmodel](https://github.com/epfml/sent2vec) and then follow the following steps:

git clone https://github.com/epfml/sent2vec.git
cd sent2vec
make
pip install . 

Now we need to download the weights for the BioSentVec from here[BioSentVecweights](https://drive.google.com/file/d/1yAUfKUVKZlsMarOTgHt_8QdRbYSxiAYa/view?usp=sharing)

Now change the path to the relative folder where you have stored the weights in the biosentvec.py file and run the script.

Also note that the code uses an inbuilt customized python class called as clusters, these have to run as well before running the biosentvec.py script.



## Set up the covbert_20000 model 

The weights can be downloaded for the 20000 data points in the SQUAD dataset at [Weightsforcovbert](https://drive.google.com/drive/folders/1MEkQAQacoEQt4jQNbs3tR2uY4SL-Mkn2?usp=sharing)

all you need to do is change the path to the location of the weights file and run the covid_qna.py script.

Te results of previous experiments have been stored in the resuklts.txt file.