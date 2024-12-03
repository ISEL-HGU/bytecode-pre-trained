import os
import random

def list_random_files(input_dir, output_file, num_files, extensions=None):
    
    all_files = []

    
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if extensions:
                if not any(file.lower().endswith(ext) for ext in extensions):
                    continue
            file_path = os.path.join(root, file)
            all_files.append(file_path)

    
    if len(all_files) > num_files:
        selected_files = random.sample(all_files, num_files)
    else:
        selected_files = all_files  

    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for file_path in selected_files:
            print(file_path)
            f_out.write(f"{file_path}\n")


input_directory = '' # put directory ex) ./files/test
data_file = '' # put your output txt file / it will be an input of pretrain_byteT5 and pretrain_byteBERT
file_extensions = ['.txt']  
number_of_files = 12000  # put total number of files ex) train-40000 test-12000 validation-8000

list_random_files(input_directory, data_file, number_of_files, extensions=file_extensions)


