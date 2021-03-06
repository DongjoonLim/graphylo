# Graphylo
Graphylo is a Deep-Learning model that aims to extract useful information from evolutionary genome data. In order to run Graphylo, the genome alignment data of different species with their reconstructed ancestral sequences are needed. You also need .bed files indicating the hg38 coordinates that corresponds to the training data you want to make.

## Downloading and Preprocessing alignment data.
1. Start with making necessary repositories. $mkdir data   $mkdir graphs
2. Download the sequence alignment data(.maf) to the data repository. You can download them from http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/
3. Run the parserPreprocess.py to convert maf file to npy file that is easier to preprocess training sets. The resulting npy file will be saved in the repository.
4. Run the graphPreprocess_tfbinding_withinCell.py to extract orthologous regions of the training set you want to be in your training set. It takes in .bed file that contains all the coordinates based on hg38 assembly. The arguments the script takes are Transcription Factor, Celltype, Chromosome
* example) python3 graph_preprocess_tfbinding_withinCell.py CTCF PC-3 2 
5. Run the preprocessRevComp.py to concatenate reverse complement to the original training set.

## Training graphylo
6. Run train_graphylo_siamese.py to train graphylo with the graphs you have made previously.
