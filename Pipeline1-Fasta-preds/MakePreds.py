# # Instruction for Prott5 embedding



from embedding import Embedding
from Bio import SeqIO
import random
import copy
import joblib
import numpy as np
import pandas as pd
import pickle
import sys



def start_embedding(inputFile=None, type='protein'):
    """
    Starts the embedding process.

    Args:
        inputFile (str): Path to the input file. Default is None.
        type (str): Type of embedding. Default is 'protein'.

    Returns:
        None
    """
    try:
        EMBEDDING_INPUT_FILE = f'./data/{inputFile}.fasta'
        EMBEDDING_OUTPUT_NAME = f'blind_{inputFile}'  # output file name

        EMBEDDING_OUTPUT_DIR = "data"  # output file directory, remember don't put / at the end
        Embedder = Embedding(
            in_file=EMBEDDING_INPUT_FILE,
            out_name=EMBEDDING_OUTPUT_NAME,
            out_dir=EMBEDDING_OUTPUT_DIR,
            level=type,  
            embed='prott5'
        )

        Embedder.embedding()
    except FileNotFoundError as e:
        print("Input file not found. Make sure the file is in the data folder.")
        sys.exit(1)

    except Exception as e:
        print('Error: ',e)  
        sys.exit(1)

def make_preds(inputFile):
    """
    Generate predictions for protein properties using pre-trained models.
    Kcat and Sc/o predictions are generated and saved to a CSV file.

    Args:
        inputFile (str): The name of the input file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input file or pre-trained models are not found.

    """
    try:
        df = pd.read_csv(f"./data/blind_{inputFile}_protein_prott5.csv")
        proteinId = df['ProteinID']
        embeddings = df.drop(columns=['ProteinID'])
        kcat = joblib.load(r"./ridgeModels/kcat mean 0.01.pkl")
        sco = joblib.load(r"./ridgeModels/Sco mean 0.01.pkl")
        kcat_predictions = kcat.predict(embeddings)
        sco_predictions = sco.predict(embeddings)

        kcat_predictions = pd.DataFrame(kcat_predictions, columns=['Kcat'])
        sco_predictions = pd.DataFrame(sco_predictions, columns=['Sc/o'])
        new_df = pd.concat([proteinId, kcat_predictions, sco_predictions], axis=1)
        new_df.to_csv(f"./data/{inputFile}-predictions.csv", index=False)
        print(new_df.shape)
        print(f"Output File: ./data/{inputFile}-predictions.csv")

    except Exception as e:
        print('Error: ',e)  
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py input_file_name \n#NOTE: Make sure to put the file in the data folder")
        sys.exit(1)
    print("Returns predictions for the input file. The output will be saved in the data folder \n#NOTE: Make sure to put the file in the data folder")


    inputFile = sys.argv[1]  #NOTE: Make sure to put the file in the data folder
    start_embedding(inputFile)
    make_preds(inputFile)