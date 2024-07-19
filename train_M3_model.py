import os
import typer
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
import time
import threading
from scipy.spatial.transform import Rotation

#tokenizer imports
from tokenizers import Tokenizer, pre_tokenizers, Regex, normalizers, decoders
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Replace
from tokenizers.pre_tokenizers import WhitespaceSplit, Split
from tokenizers.trainers import BpeTrainer, WordLevelTrainer

#transformer imports
from transformers import (
    PreTrainedTokenizerFast, 
    DataCollatorForLanguageModeling, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    GPT2Config, 
    GPTNeoConfig,
    GPTJConfig, 
    GPTNeoModel
)

#environemnt variables
os.system('clear')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def load_ldr_data(ldr_dir: Path, num_augments_per_file, is_eval_set):
    """
    Loads all LDR files from the ldr_dir and calls process_file() to augement them
    """
    src_files = list(ldr_dir.glob("*.ldr")) + list(ldr_dir.glob("*.mpd"))
    all_lines = []

    #setting upper limit for augmenting evaluation dataset
    if is_eval_set: 
        print("\n--- LOADING EVAL DATA ---")
        num_augments_per_file = min(num_augments_per_file, 1)
    else: print("--- LOADING TRAINING DATA ---")

    #iterating through files and augmenting
    for src_file in src_files:
        #progress update
        if len(all_lines) != 0 and len(all_lines) % 10000 == 0: 
            print(f"Processed {len(all_lines)} files,   latest file: {src_file.name}")
        
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

    print(f"Total Files: {len(all_lines)}")
    return all_lines

def process_file(file_lines, all_lines, num_augments_per_file, label_token, bricks_per_window=80):
    """
    Cleans up LDR by:
    -removing metadata/comments
    -rounds floats to 3 decimal places
    -adds EOS token and label token
    Creates multiple versions of a file:
    -shuffles brick lines each time
    -adds translation for entire assembly each time
    """

    for i in range(num_augments_per_file):
        processed_file_lines = []
        #determining transation for file augment
        if i == 0: translation = (0, 0, 0)
        else: translation = (random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30))
        
        for line in file_lines:
            entries = line.split()
            if entries[0] != '1': continue
            #process translation for x, y, z and rounding to 3 decimal places (inds: 2-4)
            for j in range(2, 5):
                coord = float(entries[j]) + translation[j-2]
                if coord == int(coord): entries[j] = str(int(coord))
                else: entries[j] = f"{coord:.3f}"
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
                if quat == int(quat): quaternions[j] = str(int(quat))
                else: quaternions[j] = f"{quat:.3f}"

            #creating and adding processed line to curr file lines
            final_entries = entries[0:5] + quaternions + entries[14:]
            processed_file_lines.append(" ".join(final_entries))

        #shuffling the brick lines
        random.shuffle(processed_file_lines)
        
        for i in range(len(processed_file_lines)):
            curr_window = []
            #if more then bricks_per_window bricks left get 100 brick window
            if i + bricks_per_window < len(processed_file_lines): 
                curr_window = processed_file_lines[i:i+bricks_per_window] 
            #if less then bricks_per_window bricks left get remaining and add EOS token
            else: 
                curr_window = processed_file_lines[i:]
                curr_window.append(" <|EOS|>")
            #if first window of sample, add the class label
            if i == 0: curr_window.insert(0, label_token)

            #add curr window to all lines for training data
            all_lines.append("\n".join(curr_window))
        
def load_tokenizer(vocab_size, train_lines, save_path, max_context_window=2048):
    """
    Initialises, Trains, Saves and Returns Tokenizer
    """
    m3 = Tokenizer(BPE(unk_token="<|UNK|>")) 

    #normalisation
    m3.normalizer = normalizers.Sequence([
        # Replace(Regex(r'^.*?\K\s'), " <|COL|> "), #removing colour flag to reduce token sequence length
        Replace(Regex(r'^(?:[^\s]*\s){1}[^\s]*\K\s'), " <|POS|> "),
        Replace(Regex(r'^(?:[^\s]*\s){5}[^\s]*\K\s'), " <|ORI|> "),
        Replace(Regex(r'^(?:[^\s]*\s){10}[^\s]*\K\s'), " <|SHP|> "), 
    ])

    #pretokenisation (whitespace, -, decimal places)
    m3.pre_tokenizer = pre_tokenizers.Sequence([
        WhitespaceSplit(),
        Split(pattern=Regex(r"-|\.\d{3}"), behavior="isolated"),
    ])
    
    #training tokenizer
    print("\n--- TRAINING TOKENIZER ---")
    m3_trainer = BpeTrainer(
        vocab_size = vocab_size,
        show_progress = True,
        special_tokens = [
            "<|UNK|>", "<|EOS|>", "<|PAD|>", #unk, end of sequence, padding
            "<|COL|>", "<|POS|>", "<|ORI|>", "<|SHP|>", #colour, position, orientation, shape
            "<|CR|>", "<|BU|>", "<|NA|>", "<|VE|>", #creature, building, nature, vehicle
        ] 
    )
    m3.train_from_iterator(train_lines, m3_trainer)

    # removing pre-exisiting files in tokenizer_save_path
    print(f"\n--- SAVING TOKENIZER ---\nFinal Vocab Size: {len(m3.get_vocab())}\nlocation: {save_path}\n")
    for file_name in os.listdir(save_path):
        os.remove(os.path.join(save_path, file_name))

    #converting tokenizer to transformer tokenizer and saving
    M3 = PreTrainedTokenizerFast(tokenizer_object=m3)
    M3.model_max_length = max_context_window
    M3.add_special_tokens({'pad_token': '<|PAD|>'})
    M3.add_special_tokens({'eos_token': '<|EOS|>'})
    M3.save_pretrained(save_path)
    return M3

def output_to_LDR(tokenizer, encoded_output, printTokens=False):
    """
    This function takes encoded output (raw tokens) and converts them to valid LDR by:
    -decoding tokens
    -removing special tokens
    -removing space character between '-' and the following digit
    -adding new lines after each brick line (.dat)
    """

    #printing the tokens in string format
    if printTokens:
        print(f"TOKENS:")
        # encoded_output = m3_tokenizer.encode(train_lines[2])
        str_tokens = [tokenizer.decode(tok) for tok in encoded_output]
        for tok in str_tokens:
            if ".dat" not in tok: print(tok, '|', end=" ")
            # if ".dat" not in tok: print(tok, end=" ")
            else: print(tok, end="\n")
        print("\n")

    #decoding & post-processing tokens
    string_decoding = tokenizer.decode(encoded_output)
    processed, i = "", 0    
    while i < len(string_decoding):
        ltr = string_decoding[i]
        #-
        if ltr == '-':
            processed += '-'
            i += 2
        #.dat
        elif ltr == 't':
            processed += 't\n'
            i += 2
        #special tok
        elif ltr == '<':
            i += 8
        # normal
        else:
            processed += ltr
            i += 1

    return processed

class LDRTextDataset(Dataset):
    def __init__(self, lines, tokenizer):
        self.examples = tokenizer.batch_encode_plus(lines).input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

def load_model_with_timer(config):
    """
    This function is a super complicated way of loading a model with a timer
    """
    if config == "GPT_NEO": 
        print("\n--- LOADING GPT-NEO ~16s ---")
        Config = GPTNeoConfig()
    elif config == "GPT_J": 
        print("\n--- LOADING GPT-J ~72s ---")
        Config = GPTJConfig()
    else: 
        print("\n--- LOADING GPT2 ~3s ---")
        Config = GPT2Config()
    def display_stopwatch(start_time):
        while not stop_event.is_set():
            elapsed_time = time.time() - start_time
            print(f"Elapsed Time: {elapsed_time:.2f} seconds", end="\r")
            time.sleep(0.1)
    start_time, stop_event = time.time(), threading.Event()
    thread = threading.Thread(target=display_stopwatch, args=(start_time,))
    thread.start()
    #-------ACTUALLY LOADING THE MODEL START
    model = AutoModelForCausalLM.from_config(Config)
    #-------ACTUALLY LOADING THE MODEL END
    stop_event.set()
    thread.join()
    print(f"Total Elapsed Time: {(time.time() - start_time):.2f} seconds")
    print(f"Completed. Num trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    return model

def theMain(
    #high level
    model_config: str = "GPT_2", #OPTIONS: GPT_NEO (1.3b), GPT_J (6b), GPT_2 (124m)
    vlads_device: bool = False,
    num_augments_per_file: int = 1, 
    #paths
    train_data_path: Path = Path("data"),
    save_tokenizer_path: Path = Path("trained_tokenizer"),
    save_model_path: Path = Path("trained_model"),
    #tokenizer params
    vocab_size: int = 52000,
    #model params
    num_train_epochs: int = 5,
    learning_rate: float = 1e-5,
    per_device_train_batch_size = 1,
    eval_steps: int = 10000,
    logging_steps: int = 1000,
):
    #load training data (each element = string of whole file)
    train_lines = load_ldr_data(train_data_path / 'train', num_augments_per_file, is_eval_set=False)
    eval_lines = load_ldr_data(train_data_path / 'test', num_augments_per_file, is_eval_set=True)

    #load & train tokenizer
    m3_tokenizer = load_tokenizer(vocab_size, train_lines, save_tokenizer_path)

    #tokenize data & put in tensor format
    print("--- CONVERTING DATA TO TENSORS ---")
    print("Can take a while based on dataset size...")
    train_dataset = LDRTextDataset(train_lines, m3_tokenizer)
    eval_dataset = LDRTextDataset(eval_lines, m3_tokenizer)

    #loading model and data collator
    model = load_model_with_timer(model_config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=m3_tokenizer, mlm=False)

    #setting training variables and training model
    print("\n--- TRAINING TRANSFORMER ---")
    training_args = TrainingArguments(
        output_dir=save_model_path,
        #learning
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        eval_steps=eval_steps,    
        logging_steps=logging_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        #train location
        fp16=vlads_device,
        #non-variable
        save_steps=10000,
        save_total_limit=1,
        push_to_hub=False,
        eval_strategy="steps",
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        metric_for_best_model="loss",        
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    model.save_pretrained(Path(save_model_path, model_config))
    print("\n--- TRAINED MODEL SAVED ---")

#DEAR ZACH: this is my weird way of running the main function bc I was sick of the super long terminal error messages that would cut off
import traceback
def main():
    try: theMain()
    except Exception as e: traceback.print_exception(type(e), e, e.__traceback__, limit=0)
if __name__ == "__main__": typer.run(main)

###---CHECKING FOR NUM LINES EXCEEDING MAX WINDOW SIZE---###
# count = 0
# for i in range(len(train_lines)):
#     if i % 1000 == 0: print(f"Processed {i} lines")
#     tokens = m3_tokenizer.encode(train_lines[i])
#     if len(tokens) > 2048: count += 1
# print(f"Number of lines exceeding max context window: {count} / {len(train_lines)}")
# return

###---INSPECTING TOKENS---###
# encoded_output = m3_tokenizer.encode(train_lines[0])
# str_tokens = [m3_tokenizer.decode(tok) for tok in encoded_output]
# for tok in str_tokens:
#     if ".dat" not in tok: print(tok, '|', end=" ")
#     # if ".dat" not in tok: print(tok, end=" ")
#     else: print(tok, end="\n")
# return