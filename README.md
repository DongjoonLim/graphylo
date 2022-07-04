# graphylo

1. Start with making necessary repositories. $mkdir data   $mkdir graphs
2. Download the sequence alignment data(.maf) to the data repository. You can download them from http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/
3. Run the parserPreprocess.py to convert maf file to npy file that is easier to preprocess training sets. The resulting npy file will be saved in the repository.
4. Run the graphPreprocess_withinCell.py to extract orthologous regions of the training set you want to be in your training set. It takes in .bed file that contains all the coordinates based on hg38 assembly.
5. Run the preprocessRevComp.py to concatenate reverse complement to the original training set.
6. Run train_graphylo_siamese.py to train graphylo with the graphs you have made previously.
