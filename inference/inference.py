from pathlib import Path
import os
import torch
import random
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
from transformers import PreTrainedTokenizerFast
import typer
from scipy.spatial.transform import Rotation

def decode_v1(output, tokenizer):
    tokens, ldr_lines, line, text = [], [], [], ""
    
    #creating lines
    for token_id in output[0]: tokens.append(tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False, clean_up_tokenization_spaces=False))
    for token in tokens:
        if token == "\n" or token[0] == "<":
            line.append(token)
            ldr_lines.append(line)
            line = []
        else: line.append(token)
        
    #formatting lines
    for ldr_line in ldr_lines:
        #class token
        if ldr_line[0][0] == "<": 
            text += ldr_line[0] + "\n"
            continue

        #normal brick line
        for i in range(len(ldr_line)):
            token = ldr_line[i]
            next = "" if i != len(ldr_line) else ldr_line[i+1]
            #1_
            if i == 0 and ldr_line[i] == '1': text += '1' + " "
            #colour_
            elif i == 1: text += token + " "
            #potential -
            elif ldr_line[i] == "-": text += "-"    
            #(-)int(.000)
            elif '.' in next: text += token
            #.000 or .dat
            elif '.' in token: text += token + " "
            #\n
            else: text += token   

    return text

def decode_v3(output, tokenizer):
    tokens, ldr_lines, line, text = [], [], [], ""
    
    #creating lines
    for token_id in output[0]: tokens.append(tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False, clean_up_tokenization_spaces=False))
    for token in tokens:
        if token == "\n" or token[0] == "<":
            line.append(token)
            ldr_lines.append(line)
            line = []
        else: line.append(token)
        
    #formatting lines
    for ldr_line in ldr_lines:
        #class token
        if ldr_line[0][0] == "<": 
            text += ldr_line[0] + "\n"
            continue

        #normal brick line
        for i in range(len(ldr_line)):
            token = ldr_line[i]
            next = '' if i == len(ldr_line)-1 else ldr_line[i+1]
            #1_
            if i == 0 and token == '1': text += '1' + " "
            #colour_ 
            elif i==1: text += token + ' '
            #potential -
            elif token == "-": text += "-"    
            #(-)int(.000) or \n
            elif (next != '' and next[0] == '.') or token=='\n': text += token
            #.000 or .dat or x y z
            else: text += token + " "

    return text

def valid_ldr(text):
    ldr = ""
    lines = text.split("\n")

    for line in lines:
        entries = line.split(" ")
        #class token
        if entries[0] == '': continue

        if entries[0][0] == "<":
            ldr += '0 ' + entries[0] + "\n"
            continue

        #invalid num entries
        if len(entries) != 11: 
            print(f"invalid line: {line}")
            continue
        
        #if valid line
        quats = [float(entries[5]), float(entries[6]), float(entries[7]), float(entries[8])]
        rotation = Rotation.from_quat(quats)
        rot_matrix = rotation.as_matrix().tolist()
        rot_matrix = rot_matrix[0] + rot_matrix[1] + rot_matrix[2]

        for i in range(len(rot_matrix)):
            float_val = float(rot_matrix[i])
            rot_matrix[i] = f"{float_val:.6f}"

        ldr += " ".join(entries[0:5] + rot_matrix + entries[9:]) + "\n"
    
    print(ldr)
    return ldr

def load_ldr_data(ldr_dir: Path, num_augments_per_file, is_eval_set):
    """
    Loads all LDR files from the ldr_dir and calls process_file() to augement them
    """
    src_files = list(ldr_dir.glob("*.ldr")) + list(ldr_dir.glob("*.mpd"))
    all_lines = []

    #setting upper limit for augmenting evaluation dataset
    if is_eval_set: 
        print("\n--- LOADING EVAL LINES ---")
        num_augments_per_file = min(num_augments_per_file, 1)
    else: print("--- LOADING TRAIN LINES ---")

    #iterating through files and augmenting
    for src_file in src_files:
        #determing class label
        label_token = "<|ANY|>"
        if 'creature' in src_file.name: label_token = "<|CR|>"
        elif 'building' in src_file.name: label_token = "<|BU|>"
        elif 'nature' in src_file.name: label_token = "<|NA|>"
        elif 'vehicle' in src_file.name: label_token = "<|VE|>"
        
        #augmenting & appending file
        with src_file.open('r') as file:
            file_lines = file.readlines()
            process_file(file_lines, all_lines, num_augments_per_file, label_token)

    print(f"Total Lines: {len(all_lines)}")
    return all_lines

def process_file(file_lines, all_lines, num_augments_per_file, label_token, bricks_per_window=75):
    #worst case 25 tokens per brick: 2048 / 25 = 81 bricks, rounding to 75 for extra safety
    """
    Cleans up LDR by:
    -removing metadata/comments, rounds floats to 3 decimal places
    -adds EOS token and label token
    -converts rotation matrix to quaternions
    Creates multiple versions of a file:
    -shuffles brick lines (except first pass)
    -adds translation for entire assembly (except first pass)
    """

    processed_file_lines = []
    #determining transation for file augment
    
    for line in file_lines:
        entries = line.split()
        if len(entries) == 0: continue
        if entries[0] != '1': continue
        # Round LDU x,y,z unit offset from origin to LDU integer unit
        for j in range(2, 5):
            coord = float(entries[j])
            entries[j] = str(round(coord))
        #converting rotation matrix to quaternions
        rot_matrix = [
            [float(entries[5]), float(entries[6]), float(entries[7])],
            [float(entries[8]), float(entries[9]), float(entries[10])],
            [float(entries[11]), float(entries[12]), float(entries[13])]
        ]
        rotation = Rotation.from_matrix(rot_matrix)
        quaternions = rotation.as_quat().tolist()
        for j in range(len(quaternions)):
            quat = float(quaternions[j])
            quaternions[j] = f"{quat:.3f}"

        #creating and adding processed line to curr file lines
        final_entries = entries[0:5] + quaternions + entries[14:]
        final_processed_line = " ".join(final_entries) + " \n"
        processed_file_lines.append(final_processed_line)
    
    all_lines.append(" ".join([label_token] + processed_file_lines))
     
def main(
    vlads_machine: bool = False,
    trained_tokenizer_dir: Path = Path('tokenizers/tokenizerv3'),
    trained_model_dir: Path = Path('models/modelv3/final_model'),
):
    
    #Loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(trained_tokenizer_dir)
    print(f"Tokenizer Loaded: {trained_tokenizer_dir.name}\n")
    
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not vlads_machine and torch.backends.mps.is_available(): torch.device("mps")
    model = AutoModelForCausalLM.from_pretrained(trained_model_dir).to(device)
    print(f"Model Loaded: {trained_model_dir.name}\n")

    #PROMPT AND CONFIG
    text_prompt = """<|VE|>"""

    #IF WANT TO PROMPT WITH BRICKS NEED TO ENSURE THEY ARE IN SAME FORMAT AS WHEN TRAINING
    # test_assemblies = load_ldr_data(Path("data/test"), 1, True)
    # rand_index = random.randint(0, len(test_assemblies) - 1)
    # print('potential prompt: \n', test_assemblies[rand_index])

    prompt = torch.as_tensor([tokenizer.encode(text_prompt)]).to(device)
    generation_config = GenerationConfig(
        max_length=model.config.n_positions,
        max_new_tokens=1000, #(2048 - len(prompt_tokens)),
        do_sample=True,
        top_k=5,# Can mess around with this
        top_p=0.8,# Can mess around with this
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    print("GENERATING OUTPUT...\n")
    out = model.generate(prompt, generation_config=generation_config)
    
    #PRINTING OUTPUT 
    print("\nRaw Tokens:")
    for token_id in out[0]:
        decoded_token = tokenizer.decode(token_id.unsqueeze(0), skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded_token[0] == "<": print(f"{decoded_token}\n")
        elif decoded_token == "\n": print("\\n")
        else: print(f"{decoded_token} |", end="")   
    
    #decoding tokens
    formatted_output = decode_v3(out, tokenizer)    

    #CONVERTING TO VALID LDR
    print("\nFINAL LDR:")
    ldr = valid_ldr(formatted_output)

    #saving ldr in .ldr file
    with open("output.ldr", "w") as f:
        f.write(ldr)
    
if __name__ == "__main__":
    typer.run(main)