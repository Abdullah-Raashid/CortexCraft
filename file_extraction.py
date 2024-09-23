from tqdm import tqdm
import lzma
import os

def xz_files_in_dir (directory):
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(".xz") and os.path.isfile(os.path.join(directory, filename)):
            files.append(filename)
    return files

folder_path = "/Users/abdullahraashid/Documents/PythonML/CortexCraft/openwebtext"
output_file_train = "output_train.txt"
output_file_test = "output_test.txt"
vocab_file = "vocab.txt"

files = xz_files_in_dir(folder_path)

total_files = len(files)

# Calculate the split indices
split_index = int(total_files *0.9) # 90% for training
files_train = files[:split_index]
files_test = files[split_index:]

vocab = set()

# Processing the training files
with open (output_file_train, "w", encoding = "utf-8") as outfile:
    for filename in tqdm (files_train, total = len(files_train)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt",encoding = "utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)


# Process the testing files
with open (output_file_test, "w", encoding = "utf-8") as outfile:
    for filename in tqdm (files_test, total = len(files_test)):
        file_path = os.path.join(folder_path, filename)
        with lzma.open(file_path, "rt",encoding = "utf-8") as infile:
            text = infile.read()
            outfile.write(text)
            characters = set(text)
            vocab.update(characters)

# Write the Vocabulary to vocab.txt
with open (vocab_file, "w", encoding = "utf-8") as vfile:
    for char in vocab:
        vfile.write(char + '\n')