import os
import fileinput

def fix_ids_inplace(directory):
    for filename in os.listdir(directory):
        if filename.endswith((".fasta", ".fa")):
            filepath = os.path.join(directory, filename)
            with fileinput.FileInput(filepath, inplace=True) as f:
                for line in f:
                    if line.startswith(">"):
                        print(line.replace(".", "-"), end="")
                    else:
                        print(line, end="")
            print(f"Updated: {filename}")

fix_ids_inplace("/home/chenjingjing/Models/REDfold1/redfold/data/bprna_1m/bprna_new")