# # Instruction for Prott5 embedding

# In[14]:


from embedding import Embedding
from Bio import SeqIO
import random
import copy
import joblib
import numpy as np
import pandas as pd
import pickle
import sys


"""
This code defines a list of amino acids and an exclusion list.
It also initializes an empty list to store mutated sequences.
"""

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] # XZBJ added 
exclusion_list = ['B', 'Z', 'J', 'O', 'U', 'X'] # XZBJ added
mutated_sequences = []




def start_embedding(iter, inputFile = None, j = -1, temp = False):
    """
    Starts the embedding process.

    Parameters:
    - iter (int): The iteration number.
    - inputFile (str, optional): The input file path. If not provided, a default file path will be used.
    - j (int, optional): The value of j. If -1, a default value will be used.
    - temp (bool, optional): Whether to use a temporary output name. Default is False.

    Returns:
    None
    """

    if j != -1:
        EMBEDDING_INPUT_FILE = f'./data/mutated_iter{iter}-{j}.fasta'  # input fasta file + its directory
    else:
        EMBEDDING_INPUT_FILE = f'./data/mutated_iter{iter}.fasta'  # input fasta file + its directory

    if inputFile is not None:
        EMBEDDING_INPUT_FILE = inputFile

    if temp:
        EMBEDDING_OUTPUT_NAME = f'temp_blind'
    elif j != -1:
        EMBEDDING_OUTPUT_NAME = f'iter{iter+1}-{j}_blind'
    else:
        EMBEDDING_OUTPUT_NAME = f'iter{iter+1}_blind'  # output file name
    EMBEDDING_OUTPUT_DIR = "data"  # output file directory, remember don't put / at the end
    Embedder = Embedding(
        in_file=EMBEDDING_INPUT_FILE,
        out_name=EMBEDDING_OUTPUT_NAME,
        out_dir=EMBEDDING_OUTPUT_DIR,
        level='protein',  
        embed='prott5'
    )

    Embedder.embedding()
    
    if j != -1:
        EMBEDDING_INPUT_FILE  = f'./data/mutated_iter{iter}-{j}.fasta' # input fasta file + its directory
    else:
        EMBEDDING_INPUT_FILE  = f'./data/mutated_iter{iter}.fasta' # input fasta file + its directory
    
    if inputFile != None:
        EMBEDDING_INPUT_FILE  = inputFile
    
    if temp:
        EMBEDDING_OUTPUT_NAME  = f'temp_blind'
    elif j != -1:
        EMBEDDING_OUTPUT_NAME = f'iter{iter+1}-{j}_blind'
    else:
        EMBEDDING_OUTPUT_NAME = f'iter{iter+1}_blind' # output file name
    EMBEDDING_OUTPUT_DIR = "data" # output file directory, remember don't put / at the end
    Embedder = Embedding(in_file=EMBEDDING_INPUT_FILE,
                         out_name = EMBEDDING_OUTPUT_NAME, 
                         out_dir = EMBEDDING_OUTPUT_DIR,
                         level='protein', # change to "protein" if want to embed in protein level
                         embed='prott5')

    Embedder.embedding()




def single_replacement(sequence, iterations):
    """
    Perform single replacement mutation on a given sequence.

    Args:
        sequence (Sequence): The input sequence to mutate.
        iterations (int): The number of iterations to perform the mutation.

    Returns:
        list: A list of mutated sequences.

    """
    result = []
    data = sequence.seq
    desc = sequence.description
    new_seq = sequence
    for i in range(iterations):
        new_seq = copy.deepcopy(sequence)
        
        # pick a random position
        pos = random.randint(0, len(data)-1)
        # amino acid list without the current amino acid

        removed_aa = amino_acids.copy()
        if data[pos] not in ['X' ,'Z', 'B','J']:
            removed_aa.remove(data[pos])
        # pick a random base
        new_base = random.choice(removed_aa)
        # replace the base at that position
        # print("Replacing", data[pos], "with", new_base, "at index", pos)
        mutated_data = data[:pos] + new_base + data[pos+1:]

        new_desc = desc + f"|_{i+1}-SR-{data[pos]}{pos}{new_base}"
        new_seq.seq = mutated_data
        new_seq.description = new_desc
        result.append(new_seq)
    
    return result
    




def double_replacement(sequence, iterations):
    """
    Perform double replacement mutation on a given sequence.

    Args:
        sequence (Sequence): The input sequence to mutate.
        iterations (int): The number of iterations to perform the mutation.

    Returns:
        list: A list of mutated sequences.

    """
    result = []
    data = sequence.seq
    desc = sequence.description
    new_seq = sequence
    for i in range(iterations):
        new_seq = copy.deepcopy(sequence)
        new_desc = desc + f"|_{i+1}-DR"
        # pick a random position
        pos1 = random.randint(0, len(data)-1)
        pos2 = random.randint(0, len(data)-1)
        while pos1 == pos2:
            pos2 = random.randint(0, len(data)-1)

        for pos in [pos1, pos2]:
            # amino acid list without the current amino acid
            removed_aa = amino_acids.copy()
            if data[pos] not in ['X' ,'Z', 'B','J']:
                removed_aa.remove(data[pos])
            # pick a random base
            new_base = random.choice(removed_aa)
            # replace the base at that position
    
            #print("Replacing", data[pos], "with", new_base, "at index", pos)
            new_desc += f"-{data[pos]}{pos}{new_base}"
            mutated_data = data[:pos] + new_base + data[pos+1:]

        new_seq.seq = mutated_data
        new_seq.description = new_desc
        result.append(new_seq)

    return result




def swap(sequence, iterations):
    """
    Swaps two random positions in the given sequence for the specified number of iterations.

    Args:
        sequence (Sequence): The input sequence to perform the swap operation on.
        iterations (int): The number of times to perform the swap operation.

    Returns:
        list: A list of Sequence objects, each representing a swapped version of the input sequence.
    """
    result = []
    data = sequence.seq
    desc = sequence.description
    new_seq = sequence
    for i in range(iterations):
        new_seq = copy.deepcopy(sequence)
        new_data = data
        # pick a random position
        pos1 = random.randint(0, len(data)-1)
        pos2 = random.randint(0, len(data)-1)
        while pos1 == pos2:
            pos2 = random.randint(0, len(data)-1)
        #print("Swapping", new_data[pos1], "and", new_data[pos2], "at indices", pos1, "and", pos2)
        new_desc = desc + f"|_{i+1}-S-{new_data[pos1]}{pos1}{new_data[pos2]}{pos2}"
        new_data = new_data[:pos1] + new_data[pos2] + new_data[pos1+1:]
        new_data = new_data[:pos2] + new_data[pos1] + new_data[pos2+1:]

        new_seq.seq = new_data
        new_seq.description = new_desc
        result.append(new_seq)

    return result



def mutate_sequences(iterations, file, j=-1):
    """
    Mutates the sequences in a FASTA file by performing single replacement, double replacement, and swapping operations.

    Parameters:
    - iterations (int): The number of iterations for mutation.
    - file (str): The path to the input FASTA file.
    - j (int, optional): The value used for naming the output file. Defaults to -1.

    Returns:
    - None

    """
    mutated_sequences = []  # empty the list
    fasta_file = file
    # Read the FASTA file
    sequences = SeqIO.parse(fasta_file, "fasta")

    for sequence in sequences:
        # Access the sequence ID and sequence data
        sequence_desc = sequence.description
        sequence_data = sequence.seq

        mutated_sequences.extend(single_replacement(sequence, 20))
        mutated_sequences.extend(double_replacement(sequence, 20))
        mutated_sequences.extend(swap(sequence, 10))

    # Specify the path for the new FASTA file
    if j == -1:
        output_file = f"./data/mutated_iter{iterations}.fasta"
    else:
        output_file = f"./data/mutated_iter{iterations}-{j}.fasta"

    # Write the sequences to the new FASTA file
    SeqIO.write(mutated_sequences, output_file, "fasta")
    



def grab_top(iterations, number, multi):
    """
    Reads protein embeddings from a CSV file, performs predictions using pre-trained models,
    and returns a list of the top proteins based on the 'Kcat' column.

    Parameters:
    - iterations (int): The number of iterations.
    - number (int): The number of top proteins to return.
    - multi (str): The multi value.

    Returns:
    - list: A list of the top proteins based on the 'Kcat' column.

    Example usage:
    >>> grab_top(5, 10, 'multi_value')
    ['protein1', 'protein2', 'protein3', ...]
    """
    df = pd.read_csv(f"./data/iter{iterations+1}-{multi}_blind_protein_prott5.csv")
    proteinId = df['ProteinID']
    embeddings = df.drop(columns=['ProteinID'])
    kcat = joblib.load(r"./ridgeModels/kcat mean 0.01.pkl")
    sco = joblib.load(r"./ridgeModels/Sco mean 0.01.pkl")
    kcat_predictions = kcat.predict(embeddings)
    sco_predictions = sco.predict(embeddings)

    kcat_predictions = pd.DataFrame(kcat_predictions, columns=['Kcat'])
    sco_predictions = pd.DataFrame(sco_predictions, columns=['Sc/o'])
    new_df = pd.concat([proteinId, kcat_predictions, sco_predictions], axis=1)
    new_df.to_csv(f"./data/Iter-{iterations+1}-{multi}-predictions.csv", index=False)
    print(new_df.shape)
    top_20 = new_df.nlargest(number, 'Kcat')
    top_20 = top_20['ProteinID']
    top_20 = top_20.to_list()
    top_20 = [string.replace(">", "") for string in top_20]
    
    return top_20



def grab_OG_sep_top(iterations, number, multi=-1):
    """
    Retrieves the top protein sequences based on predictions of Kcat and Sc/o values.

    Parameters:
    iterations (int): The number of iterations.
    number (int): The number of top protein sequences to retrieve.
    multi (int, optional): The multi value. Defaults to -1.

    Returns:
    list: A list of top protein sequences.

    """

    if multi == -1:
        df = pd.read_csv(f"./data/iter{iterations+1}_blind_protein_prott5.csv")
    else:
        df = pd.read_csv(f"./data/iter{iterations+1}-{multi}_blind_protein_prott5.csv")
    proteinId = df['ProteinID']
    print(proteinId.shape)

    top_ids = ['gi|313473685|dbj|BAJ40208.1|', 'gi|932247975|gb|ALG62823.1|', 'gi|932248269|gb|ALG62965.1|', 'gi|932248239|gb|ALG62950.1|', 'gi|932247944|gb|ALG62808.1|', 'gi|932247942|gb|ALG62807.1|', 'gi|932247938|gb|ALG62805.1|', 'gi|932248235|gb|ALG62948.1|', 'gi|932248233|gb|ALG62947.1|', 'gi|932247946|gb|ALG62809.1|', 'tr|A0A1C3HPS9|A0A1C3HPS9_PUCDI', 'tr|A0A1C3HPT0|A0A1C3HPT0_9POAL', 'YP_009573569.1', 'AFB70630.1', 'AGT56139.1', 'YP_899415.1', 'SCM15160.1', 'SCM15158.1', 'tr|A0A1C3HPM4|A0A1C3HPM4_9POAL', 'tr|A0A6C0SV93|A0A6C0SV93_ERATE']
    kcat = joblib.load(r"./ridgeModels/kcat mean 0.01.pkl")
    sco = joblib.load(r"./ridgeModels/Sco mean 0.01.pkl")

    top_sequences = []
    print('running Og grab')
    for i in top_ids:
        new_df = pd.DataFrame()
        for j in proteinId: 
            if i in j:
                data = df.loc[df['ProteinID'] == j]
                new_df = pd.concat([new_df, data], ignore_index=True)


        embeddings = new_df.drop(['ProteinID'], axis=1)
        
        kcat_predictions = kcat.predict(embeddings)
        sco_predictions = sco.predict(embeddings)

        kcat_predictions = pd.DataFrame(kcat_predictions, columns=['Kcat'])
        sco_predictions = pd.DataFrame(sco_predictions, columns=['Sc/o'])
        new_pred_df = pd.concat([new_df['ProteinID'], kcat_predictions, sco_predictions], axis=1)
        
        print(new_pred_df.shape)
        top_20 = new_pred_df.nlargest(number, 'Kcat') # top x number proteins with highest Kcat
        top_20 = top_20['ProteinID']
        top_20 = top_20.to_list()
        top_20 = [string.replace(">", "") for string in top_20]
        
        # add to top_sequences
        top_sequences.extend(top_20)


    embeddings = df.drop(columns=['ProteinID'])
    kcat_predictions = kcat.predict(embeddings)
    sco_predictions = sco.predict(embeddings)


    kcat_predictions = pd.DataFrame(kcat_predictions, columns=['Kcat'])
    sco_predictions = pd.DataFrame(sco_predictions, columns=['Sc/o'])
    new_df =pd. DataFrame()
    new_df = pd.concat([proteinId, kcat_predictions, sco_predictions], axis=1)

    # save the predictions of every protein
    if multi >= 0:
        new_df.to_csv(f"./data/Iter-{iterations+1}-predictions-{multi}.csv", index=False)
    else:
        new_df.to_csv(f"./data/Iter-{iterations+1}-predictions.csv", index=False)

    return top_sequences
 


def save_top_seq_fasta(iterations, top_20, multi=-1):
    """
    Save the top sequences from a FASTA file to a new FASTA file.

    Parameters:
    - iterations (int): The iteration number.
    - top_20 (list): A list of sequence IDs representing the top sequences.
    - multi (int, optional): The multi number. Defaults to -1.

    Returns:
    None
    """

    # Specify the path to the FASTA file
    if multi == -1:
        fasta_file = f"./data/mutated_iter{iterations}.fasta"
    else:
        fasta_file = f"./data/mutated_iter{iterations}-{multi}.fasta"

    # Read the FASTA file
    sequences = SeqIO.parse(fasta_file, "fasta")

    top_sequences = []
    # Iterate over the sequences
    for sequence in sequences:
        # Access the sequence ID and sequence data
        sequence_desc = sequence.description
        sequence_data = sequence.seq
        if sequence_desc not in top_20:
            continue
        # Do something with the sequence ID and sequence data
        #print(f"Sequence ID: {sequence_desc}")
        #print(f"Sequence Data: {sequence_data}")
        top_sequences.append(sequence)

    # Specify the path for the new FASTA file
    if multi >= 0:
        output_file = f"./data/top_sequences_iter{iterations+1}-{multi}.fasta"
    else:
        output_file = f"./data/top_sequences_iter{iterations+1}.fasta"

    # Write the sequences to the new FASTA file
    SeqIO.write(top_sequences, output_file, "fasta")




if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py top_sequences.fasta runs iterations_per_run")
        sys.exit(1)


    top_sequences_fasta = sys.argv[1]
    number_of_runs = sys.argv[2]
    iterations_per_run = sys.argv[3]
    
    # NOTE TLDR: Runs Genetic Algorithm to find the best sequences.
    """
    This code performs a series of operations on protein sequences to generate mutated sequences and make predictions using pre-trained models.
    The steps involved are as follows:
    1. Load pre-trained models using joblib.
    2. Iterate over a range of values.
    3. Read a FASTA file containing protein sequences.
    4. Perform various mutations on each sequence.
    5. Write the mutated sequences to a new FASTA file.
    6. Perform embedding on the mutated sequences.
    7. Make predictions using the pre-trained models.
    8. Save the predictions to a CSV file.
    9. Select the top 5 sequences based on the 'Kcat' value.
    10. Filter out the top 5 sequences from the mutated sequences.
    11. Write the filtered sequences to a new FASTA file.
    12. Perform embedding on the filtered sequences.
    13. Select the top 100 sequences based on a specific criterion.
    14. Save the top 100 sequences to a new FASTA file.
    15. Iterate over a range of values again.
    16. Mutate the sequences based on the iteration value.
    17. Perform embedding on the mutated sequences.
    18. Select the top sequences based on a specific criterion.
    19. Save the top sequences to a new FASTA file.
    """

    kcat = joblib.load(r"./ridgeModels/kcat mean 0.01.pkl")
    sco = joblib.load(r"./ridgeModels/Sco mean 0.01.pkl")

    for j in range(1,number_of_runs+1):
        print(j)
        # Load models using pickle instead of joblib
        iterations = 0

        mutated_sequences = [] #empty the list
        temp_sequences = [] #empty the list

        fasta_file = f"./data/{top_sequences_fasta}"
        # Read the FASTA file
        sequences = SeqIO.parse(fasta_file, "fasta")

        for sequence in sequences:
            temp_sequences = []
            # Access the sequence ID and sequence data
            sequence_desc = sequence.description
            sequence_data = sequence.seq

            temp_sequences.append(sequence)

            temp_sequences.extend(single_replacement(sequence,20))

            temp_sequences.extend(double_replacement(sequence,20))

            temp_sequences.extend(swap(sequence,10))

            # Write the sequences to the new FASTA file
            SeqIO.write(temp_sequences, './data/tempfile.fasta', "fasta")

            start_embedding(iterations, './data/tempfile.fasta', -1, True)

            df = pd.read_csv(f"./data/temp_blind_protein_prott5.csv")
            proteinId = df['ProteinID']
            embeddings = df.drop(columns=['ProteinID'])
            kcat_predictions = kcat.predict(embeddings)
            sco_predictions = sco.predict(embeddings)

            kcat_predictions = pd.DataFrame(kcat_predictions, columns=['Kcat'])
            sco_predictions = pd.DataFrame(sco_predictions, columns=['Sc/o'])
            new_df = pd.concat([proteinId, kcat_predictions, sco_predictions], axis=1)
            new_df.to_csv(f"./data/Iter-{iterations+1}-{j}-predictions.csv", index=False)

            top_5 = new_df.nlargest(5, 'Kcat')
            top_5 = top_5['ProteinID']
            top_5 = top_5.to_list()
            top_5 = [string.replace(">", "") for string in top_5]


            # Specify the path to the FASTA file
            temp_fasta_file = f"./data/tempfile.fasta"

            # Read the FASTA file
            tsequences = SeqIO.parse(temp_fasta_file, "fasta")

            # Iterate over the sequences
            for tsequence in tsequences:
                # Access the sequence ID and sequence data
                sequence_desc = tsequence.description
                sequence_data = tsequence.seq
                if sequence_desc not in top_5:
                    continue
                mutated_sequences.append(tsequence)

        output_file = f"./data/mutated_iter{iterations}-{j}.fasta"

        # Write the sequences to the new FASTA file
        SeqIO.write(mutated_sequences, output_file, "fasta")

        start_embedding(iterations, j=j)

        top_100 = grab_top(iterations, 100, j)

        save_top_seq_fasta(iterations, top_100, j)

        for i in range(1,iterations_per_run+1):
            print(i)
            mutate_sequences(i, f"./data/top_sequences_iter{i}-{j}.fasta", j)

            start_embedding(i, f'./data/mutated_iter{i}-{j}.fasta', j = j)

            top_seq = grab_OG_sep_top(i, 5, j)

            save_top_seq_fasta(i, top_seq, j)
