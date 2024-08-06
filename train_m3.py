import os
import typer
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation

#tokenizer imports
from tokenizers import Tokenizer, pre_tokenizers, Regex #normalizers, decoders
from tokenizers.models import BPE
from tokenizers.normalizers import Replace
from tokenizers.pre_tokenizers import Split
from tokenizers.decoders import ByteLevel
from tokenizers.trainers import BpeTrainer

#transformer imports
from transformers import (
    PreTrainedTokenizerFast, 
    DataCollatorForLanguageModeling, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    GPT2Config, 
    # GPTNeoConfig, 
)

#environemnt variables
os.system('clear')
os.environ["TOKENIZERS_PARALLELISM"] = "true"

def printTokenStream(text, m3):
    #untouched text
    # print("\n---TRAIN TEXT UNTOUCHED---\n", text, end="\n--END TRAIN TEXT UNTOUCHED--\n\n")
    print("---Sample Train Line and Tokens---\n")
    print(text, "\n")
    
    #tokenized text
    input_ids = m3.encode(text, add_special_tokens=True)
    tokens = m3.convert_ids_to_tokens(input_ids)
    for tok in tokens:
        if tok == '\n': print(r"'\n'" )
        else: print(f"'{tok}' ", end="")
    print('\n')

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

    for i in range(num_augments_per_file):
        processed_file_lines = []
        #determining transation for file augment
        if i == 0: translation = (0, 0, 0)
        else: translation = (random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30))
        
        for line in file_lines:
            entries = line.split()
            if len(entries) == 0: continue
            if entries[0] != '1': continue
            #process translation for x, y, z and rounding to 3 decimal places (inds: 2-4)
            for j in range(2, 5):
                coord = float(entries[j]) + translation[j-2]
                entries[j] = f"{coord:.3f}"
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

        #shuffling the brick lines
        if i != 0: random.shuffle(processed_file_lines)
        
        #creating training windows
        for j in range(len(processed_file_lines)):
            curr_window = []
            #if more then bricks_per_window bricks left get 100 brick window
            if j + bricks_per_window < len(processed_file_lines): 
                curr_window = processed_file_lines[j:j+bricks_per_window] 
            #if less then bricks_per_window bricks left get remaining and add EOS token
            else: 
                curr_window = processed_file_lines[j:]
                curr_window.append(" <|EOS|>")
            #if first window of sample, add the class label
            if j == 0: curr_window.insert(0, label_token)

            #add curr window to all lines for training data
            all_lines.append(" ".join(curr_window))
             
def load_tokenizer(vocab_size, train_lines, save_path, max_context_window=2048):
    """
    Initialises, Trains, Saves and Returns Tokenizer
    """
    m3 = Tokenizer(BPE(unk_token="<|UNK|>")) 

    #normalisation
    # m3.normalizer = normalizers.Sequence([
    #     Replace(Regex(r'^.*?\K\s'), " <|COL|> "), 
    #     Replace(Regex(r'^(?:[^\s]*\s){1}[^\s]*\K\s'), " <|POS|> "), 
    #     Replace(Regex(r'^(?:[^\s]*\s){5}[^\s]*\K\s'), " <|ORI|> "), 
    #     Replace(Regex(r'^(?:[^\s]*\s){10}[^\s]*\K\s'), " <|SHP|> "), 
    # ])

    #pre-tokenization (spaces, -, decimal places)
    m3.pre_tokenizer = pre_tokenizers.Sequence([
        Split(pattern=Regex(r" "), behavior="removed"),  # Splits on spaces but preserves newlines
        Split(pattern=Regex(r"-|\.\d{3}"), behavior="isolated"),
    ])
    m3.decoder = ByteLevel()
    
    #training tokenizer
    print("\n--- TRAINING TOKENIZER ---")
    m3_trainer = BpeTrainer(
        vocab_size = vocab_size,
        show_progress = True,
        special_tokens = [
            "<|UNK|>", "<|EOS|>", "<|PAD|>", #unk, end of sequence, padding
            "<|COL|>", "<|POS|>", "<|ORI|>", "<|SHP|>", #colour, position, orientation, shape
            "<|ANY|>", "<|CR|>", "<|BU|>", "<|NA|>", "<|VE|>", #creature, building, nature, vehicle
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

class LDRTextDataset(Dataset):
    def __init__(self, lines, tokenizer):
        self.examples = tokenizer.batch_encode_plus(lines).input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])

def theMain(
    #high level
    vlads_device: bool = False,
    num_augments_per_file: int = 1, 
    #paths
    train_data_path: Path = Path("data"),
    save_tokenizer_path: Path = Path("trained_tokenizer"),
    save_model_path: Path = Path("trained_model"),
):
    #load training data (each element = string of whole file)
    train_lines = load_ldr_data(train_data_path / 'test', num_augments_per_file, is_eval_set=False)
    eval_lines = load_ldr_data(train_data_path / 'test', num_augments_per_file, is_eval_set=True)

    #load & train tokenizer
    m3_tokenizer = load_tokenizer(int(52000), train_lines, save_tokenizer_path)
    printTokenStream(train_lines[0], m3_tokenizer)

    #tokenize data & put in tensor format
    print("--- CONVERTING DATA TO TENSORS ---")
    print("Can take a while based on dataset size...")
    train_dataset = LDRTextDataset(train_lines, m3_tokenizer)
    print("completed training dataset")
    eval_dataset = LDRTextDataset(eval_lines, m3_tokenizer)
    print("completed evaluation dataset")

    #loading model and data collator
    print("\n--- LOADING GPT2 and Data Collator---")
    model = AutoModelForCausalLM.from_config(GPT2Config(
        vocab_size=m3_tokenizer.vocab_size,
        n_positions=m3_tokenizer.model_max_length,
    ))
    data_collator = DataCollatorForLanguageModeling(tokenizer=m3_tokenizer, mlm=False)
    print("done")

    #setting training variables and training model
    print("\n--- TRAINING TRANSFORMER ---")
    training_args = TrainingArguments(
        #paths
        output_dir=save_model_path,
        fp16=vlads_device,

        #learning variables
        num_train_epochs=10,
        learning_rate=5e-6, # original divided 2, data is 9 times larger
        eval_steps=1000,    # 1 step = 10 samples, so every 10k samples

        #updates to weights occer every trainBatchSize * gradiatentAccumSteps = every 10 samples
        per_device_train_batch_size=2, #num samples simultaneously processed (in 1 batch)
        gradient_accumulation_steps=5, #num batches before updating weights
        
        #non-variable args
        logging_steps=1000,
        save_steps=10000,
        save_total_limit=1,
        push_to_hub=False,
        eval_strategy="steps",
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        metric_for_best_model="loss",   
        weight_decay=0.01,
        greater_is_better=False,
        lr_scheduler_type="cosine",     
    )

    #saving training args so we can see what we trained with
    if training_args_file.exists(): training_args_file.unlink()
    training_args_file = save_model_path / "trainingArgs_M3_GPT2_v1-1.txt"
    with open(training_args_file, 'w') as f:
        for arg, value in vars(training_args).items():
            f.write(f"{arg}: {value}\n")

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    print("\n--- TRANSFORMER TRAINED ---")
    model.save_pretrained(Path(save_model_path, "M3_GPT2_v1-1"))
    print("\n--- TRAINED MODEL SAVED ---")

#DEAR ZACH: this is my weird way of running the main function bc I was sick of the super long terminal error messages that would cut off
import traceback
def main():
    try: theMain()
    except Exception as e: traceback.print_exception(type(e), e, e.__traceback__, limit=0)
if __name__ == "__main__": typer.run(main)
