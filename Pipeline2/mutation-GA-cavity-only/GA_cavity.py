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



"""
This code defines a list of amino acids, an exclusion list, and a binding dictionary.
It also initializes an empty list for mutated sequences.

- amino_acids: A list of amino acids.
- exclusion_list: A list of amino acids to be excluded.
- binding_dict: A dictionary where the keys are protein identifiers and the values are lists of binding cavity positions.
- mutated_sequences: An empty list to store mutated sequences.
"""

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V'] # XZBJ added 
exclusion_list = ['B', 'Z', 'J', 'O', 'U', 'X'] # XZBJ added
binding_dict = {
    'gi|932248239|gb|ALG62950.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'gi|932248233|gb|ALG62947.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'gi|932248235|gb|ALG62948.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'gi|932247944|gb|ALG62808.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'gi|932247942|gb|ALG62807.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'gi|932247938|gb|ALG62805.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'gi|932247946|gb|ALG62809.1|': [167, 168, 169, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198],
    'gi|932248269|gb|ALG62965.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'gi|313473685|dbj|BAJ40208.1|': [111, 285, 286, 287, 303, 304, 305, 306, 307, 308, 310, 311, 312, 313, 314, 315, 316],
    'gi|932247975|gb|ALG62823.1|': [116, 290, 291, 292, 308, 309, 310, 311, 312, 313, 315, 316, 317, 318, 319, 320, 321],
    'AFB70630.1': [135, 309, 310, 311, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340],
    'tr|A0A1C3HPM4|A0A1C3HPM4_9POAL': [135, 309, 310, 311, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340],
    'tr|A0A1C3HPS9|A0A1C3HPS9_PUCDI': [135, 309, 310, 311, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340],
    'tr|A0A1C3HPT0|A0A1C3HPT0_9POAL': [135, 309, 310, 311, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340],
    'tr|A0A6C0SV93|A0A6C0SV93_ERATE': [142, 316, 317, 318, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 346, 347],
    'YP_899415.1': [135, 309, 310, 311, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340],
    'SCM15160.1': [135, 309, 310, 311, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340],
    'SCM15158.1': [135, 309, 310, 311, 327, 328, 329, 330, 331, 332, 334, 335, 336, 337, 338, 339, 340],
    'YP_009573569.1': [142, 316, 317, 318, 334, 335, 336, 337, 338, 339, 341, 342, 343, 344, 345, 346, 347],
    'AGT56139.1': [121, 295, 296, 297, 313, 314, 315, 316, 317, 318, 320, 321, 322, 323, 324, 325, 326]
}
mutated_sequences = []



def start_embedding(iter, inputFile=None, j=-1, temp=False):
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




def single_replacement(sequence, iterations):
    """
    Perform single base replacement mutation on a given sequence. Here we randomly select a position in the sequence and replace the base at that position with a random base.

    Args:
        sequence (Bio.SeqRecord.SeqRecord): The input sequence to mutate.
        iterations (int): The number of iterations to perform the mutation.

    Returns:
        list: A list of mutated sequences.

    """
    result = []
    data = sequence.seq
    desc = sequence.description

    seq_id = sequence.id
    binding_indices = binding_dict[seq_id]

    
    new_seq = sequence
    for i in range(iterations):
        new_seq = copy.deepcopy(sequence)
        
        # pick a random position (element) from the binding indices
        pos = random.choice(binding_indices)
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
    Perform double replacement mutation on a given sequence. Here we randomly select two positions in the sequence and replace the bases at those positions with random bases.

    Args:
        sequence (Bio.SeqRecord.SeqRecord): The input sequence to mutate.
        iterations (int): The number of iterations to perform the mutation.

    Returns:
        list: A list of mutated sequences.

    """
    result = []
    data = sequence.seq
    desc = sequence.description
    seq_id = sequence.id
    binding_indices = binding_dict[seq_id]

    new_seq = sequence
    for i in range(iterations):
        new_seq = copy.deepcopy(sequence)
        new_desc = desc + f"|_{i+1}-DR"
        # pick a random position (element) from the binding indices
        pos1 = random.choice(binding_indices)
        pos2 = random.choice(binding_indices)
        while pos1 == pos2:
            pos2 = random.choice(binding_indices)

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
    Swaps elements in a sequence based on the given iterations. Here we randomly select two positions in the sequence and swap the elements at those positions.

    Args:
        sequence (Sequence): The input sequence to swap elements in.
        iterations (int): The number of iterations to perform the swapping.

    Returns:
        list: A list of sequences with swapped elements.
    """
    result = []
    data = sequence.seq
    desc = sequence.description
    seq_id = sequence.id
    binding_indices = binding_dict[seq_id]
    new_seq = sequence
    for i in range(iterations):
        new_seq = copy.deepcopy(sequence)
        new_data = data
        # pick a random position (element) from the binding indices
        pos1 = random.choice(binding_indices)
        pos2 = random.choice(binding_indices)
        while pos1 == pos2:
            pos2 = random.choice(binding_indices)
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
    Mutates the sequences in a FASTA file by performing single replacement, double replacement, and swapping operations. The mutated sequences are written to a new FASTA file.

    Parameters:
    iterations (int): The number of iterations for mutation.
    file (str): The path to the input FASTA file.
    j (int, optional): The value used for creating the output file name. Defaults to -1.

    Returns:
    None
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
    Retrieves the top proteins based on predictions of Kcat values. 

    Args:
        iterations (int): The number of iterations.
        number (int): The number of top proteins to retrieve.
        multi (str): The multi value.

    Returns:
        list: A list of the top proteins based on Kcat predictions.
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
    Retrieves the top protein sequences based on predictions of Kcat and Sc/o values. but this function is used to force the original parent sequences to be included in the top proteins.

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
    if multi>=0:
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
    runs = sys.argv[2]
    iters = sys.argv[3]

    # NOTE TLDR: Runs Genetic Algorithm to find the best sequences (Only mutates in the cavity regions)
    """
    This code performs a series of operations on protein sequences to generate mutated sequences and make predictions using pre-trained models.
    The steps involved are as follows:
    1. Load pre-trained models for kcat and sco using joblib.
    2. Iterate over a range of values from 1 to 5.
    3. Read protein sequences from a FASTA file.
    4. Generate mutated sequences by performing single replacement, double replacement, and swap operations on the original sequences.
    5. Write the mutated sequences to a temporary FASTA file.
    6. Perform embedding on the temporary FASTA file using a start_embedding function.
    7. Read the resulting embeddings from a CSV file.
    8. Make predictions for kcat and sco using the pre-trained models.
    9. Combine the predictions with the protein IDs and save the results to a CSV file.
    10. Select the top 5 sequences based on the kcat predictions.
    11. Filter the mutated sequences to include only the top 5 sequences.
    12. Write the filtered sequences to a new FASTA file.
    13. Perform embedding on the filtered sequences using a start_embedding function.
    14. Select the top 100 sequences based on the embeddings.
    15. Save the top 100 sequences to a new FASTA file.
    16. Iterate over a range of values from 1 to 10.
    17. Mutate the top sequences from the previous step.
    18. Perform embedding on the mutated sequences.
    19. Select the top 5 sequences based on the embeddings.
    20. Save the top 5 sequences to a new FASTA file.
    """

    kcat = joblib.load(r"./ridgeModels/kcat mean 0.01.pkl")
    sco = joblib.load(r"./ridgeModels/Sco mean 0.01.pkl")

    for j in range(1,runs+1):
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

        for i in range(1,iters+1):
            print(i)
            mutate_sequences(i, f"./data/top_sequences_iter{i}-{j}.fasta", j)

            start_embedding(i, f'./data/mutated_iter{i}-{j}.fasta', j = j)

            top_seq = grab_OG_sep_top(i, 5, j)

            save_top_seq_fasta(i, top_seq, j)
