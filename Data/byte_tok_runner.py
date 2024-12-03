import subprocess
import os
import shlex
import re

def tokenize_class_file_with_gradle(project_dir, class_file_path, additional_args):
    gradle_cmd = ['gradle', 'run']
    
    args_list = [class_file_path] + additional_args

    
    args_str = ' '.join([shlex.quote(arg) for arg in args_list])
    gradle_cmd.append(f'--args={args_str}')

    
    process = subprocess.run(
        gradle_cmd,
        cwd=project_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    
    if process.returncode != 0:
        print(f"Error in Gradle execution: {process.stderr}")
        return None

    
    output = process.stdout
    
    
    tokens = extract_tokens_from_output(output)
    
    
    
    return tokens

def extract_tokens_from_output(output):
    start_marker = 'run'
    end_marker = 'BUILD'

    
    start_idx = output.find(start_marker)
    end_idx = output.find(end_marker, start_idx)

    if start_idx == -1 or end_idx == -1:
        print("Tokenized output markers not found.")
        return None

    tokenized_output = output[start_idx + len(start_marker):end_idx].strip()
    tokens = tokenized_output.splitlines()
    
    
    tokens = [line.strip() for line in tokens if line.strip()]
    token_groups = split_tokens_by_marker(tokens, marker='[marker]')
    
    
    return token_groups

def split_tokens_by_marker(tokens, marker='[marker]'):
    if tokens is None:
        return []

    split_groups = []
    current_group = []

    for token in tokens:
        if token == marker:
            if current_group:
                split_groups.append(current_group)
                current_group = []
        else:
            current_group.append(token)

    
    if current_group:
        split_groups.append(current_group)

    return split_groups
