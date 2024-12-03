# bytecode-pre-trained

### 1. unzip data files (1. test_files.zip 2. train.zip 3. valid.zip)

### 2. generate input paths by using generate_input_path.py
- you should put configurations on generate_input_path.py
- directory path and output text file name
- maximum number is 1) train = 40,000 2) validation = 8,000 3) test = 12,000

### 3. Put your environment on pretrain_byteT5.py and pretrain_byteBERT.py
- train.txt path
- valid.txt path
- test.txt path
- GPU environment
- save point directory

### 4. Run byteT5.py and byteBERT.py with train mode
- use python3 Model/byteT5.py --mode train

### 5. Run byteT5.py and byteBERT.py with test mode
- use python3 Model/byteT5.py --mode test --model_file "your savepoint path"
