# Graphylo
Graphylo is a Deep-Learning model that aims to extract useful information from evolutionary genome data. In order to run Graphylo, the genome alignment data of different species with their reconstructed ancestral sequences are needed. You also need .bed files indicating the hg38 coordinates that corresponds to the training data you want to make.

## Downloading and Preprocessing alignment data.
1. Start with making necessary repositories. $mkdir data   $mkdir Models 
2. create conda environment from yaml file. conda env create -f environment.yml
3. Download the sequence alignment data(.maf) to the data repository. You can download them from http://repo.cs.mcgill.ca/PUB/blanchem/Boreoeutherian/
4. Run the parserPreprocess.py to convert maf file to npy file that is easier to preprocess training sets. The resulting .pkl file will be saved in the repository.
5. make .bed file where each line is in the format 
*chr1    index  index+1  binary_label 
see data/example/bed
6. Run the preprocess_graphs.py to extract orthologous regions of the training set you want to be in your training set. It takes in .bed file that contains all the coordinates based on hg38 assembly. The arguments the script takes are path to the bed file, Chromosome, output path for the training data, output path for the training label. 
* example) python3 preprocess_graphs.py data/example.bed 22 data/example_X.npy data/example_y.npy
7. Run the preprocessRevComp.py to concatenate reverse complement to the original training set.

## Training graphylo
7. Run train_graphylo_siamese.py to train graphylo with the data you have preprocessed previously. python3 train_graphylo_siamese.py data_path output_model_path target_path gpu
* example) python3 train_graphylo_siamese.py data/example_X.npy Models/model data/example_y.npy 3

## Predicting using Graphylo
1. you can simply write model.predict(some_data) to predict data
* model_evolstm = tf.keras.models.load_model(f'models/evolstm_mutation')
* predictions_graphylo_lstm = model_evolstm.predict(examples_graphylo, batch_size=64)
