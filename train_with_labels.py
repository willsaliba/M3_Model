import os
import typer
import random
from pathlib import Path
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation
from datetime import datetime

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
    EarlyStoppingCallback,
    GPT2LMHeadModel
    # GPTNeoConfig, 
)

#environemnt variables
os.system('clear')
os.environ["TOKENIZERS_PARALLELISM"] = "true"


class LabelConditionedGPT2(GPT2LMHeadModel):
    def __init__(self, config, num_labels):
        super().__init__(config)
        
        # Define a new embedding layer for labels
        self.label_embedding = torch.nn.Embedding(num_labels, config.hidden_size)
        
        # Initialize label embeddings
        torch.nn.init.normal_(self.label_embedding.weight, mean=0, std=config.initializer_range)
    
    def forward(self, input_ids, attention_mask=None, labels=None, label_ids=None):
        # Get label embeddings
        if label_ids is not None:
            label_embeds = self.label_embedding(label_ids)  # Shape: (batch_size, hidden_size)
            # Expand label embeddings to match input length
            label_embeds = label_embeds.unsqueeze(1).expand(-1, input_ids.size(1), -1)  # Shape: (batch_size, seq_len, hidden_size)
        else:
            label_embeds = 0  # If no label, set label embedding to zero
        
        # Get text embeddings
        text_outputs = self.transformer(input_ids, attention_mask=attention_mask)
        text_embeds = text_outputs.last_hidden_state  # Shape: (batch_size, seq_len, hidden_size)
        
        # Combine label and text embeddings (simple addition here)
        combined_embeds = text_embeds + label_embeds
        
        # Pass combined embeddings through the language model head
        lm_logits = self.lm_head(combined_embeds)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'loss': loss, 'logits': lm_logits} if loss is not None else {'logits': lm_logits}

def printTokenStream(text, m3):
    print("---Sample Train Line and Tokens---\n")
    print(text, "\n")
    
    # Tokenized text
    inputs = m3(text, add_special_tokens=True, return_tensors='pt')
    tokens = m3.convert_ids_to_tokens(inputs['input_ids'][0])
    for tok in tokens:
        if tok == '\n': 
            print(r"'\n'" )
        else: 
            print(f"'{tok}' ", end="")
    print('\n')

# Returns [(text, label), ...]
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


def process_file(file_lines, all_lines, num_augments_per_file, label, bricks_per_window=75):
    """
    Cleans up LDR by:
    - Removing metadata/comments, rounds floats to 3 decimal places.
    - Adds EOS token.
    - Converts rotation matrix to quaternions.
    Creates multiple versions of a file:
    - Shuffles brick lines (except first pass).
    - Adds translation for entire assembly (except first pass).
    """

    for i in range(num_augments_per_file):
        processed_file_lines = []
        # Determine translation for file augment
        if i == 0:
            translation = (0, 0, 0)
        else:
            translation = (random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30))
        
        for line in file_lines:
            entries = line.split()
            if len(entries) == 0: continue
            if entries[0] != '1': continue
            
            # Round LDU x, y, z unit offset from origin to LDU integer unit
            for j in range(2, 5):
                coord = float(entries[j]) + translation[j-2]
                entries[j] = str(round(coord))
            
            # Convert rotation matrix to quaternions
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

            # Create and add processed line to current file lines
            final_entries = entries[0:5] + quaternions + entries[14:]
            final_processed_line = " ".join(final_entries) + " \n"
            processed_file_lines.append(final_processed_line)

        # Shuffle the brick lines
        if i != 0:
            random.shuffle(processed_file_lines)
        
        # Create training windows
        for j in range(len(processed_file_lines)):
            curr_window = []
            
            # If not the first window & fewer than 8 bricks left, stop
            if j != 0 and len(processed_file_lines) - j < 8:
                break

            # If more than bricks_per_window bricks left
            if j + bricks_per_window < len(processed_file_lines): 
                curr_window = processed_file_lines[j:j+bricks_per_window] 
            else:  # If fewer than bricks_per_window bricks left, get remaining and add EOS token
                curr_window = processed_file_lines[j:]
                curr_window.append(" <|EOS|>")
            
            # Combine window into a single text and add to all_lines
            combined_window = " ".join(curr_window)
            all_lines.append((combined_window, label))  # Add text with its corresponding label

             
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
    def __init__(self, data, tokenizer, label_map):
        """
        Initialize the LDRTextDataset.

        Args:
        - data: List of (text, label) tuples.
        - tokenizer: The tokenizer to use for encoding the data.
        - label_map: Dictionary mapping label names to label IDs.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label_map = label_map  # Mapping from label name to label ID
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get the tokenized representation of a data sample.

        Args:
        - idx: Index of the sample to retrieve.

        Returns:
        - A dictionary containing 'input_ids', 'attention_mask', 'labels', and 'label_ids'.
        """
        text, label = self.data[idx]  # Assuming data is a list of (text, label) tuples
        
        # Tokenize the text
        text_inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )

        # Convert the label to an ID using the label_map
        label_id = torch.tensor(self.label_map[label], dtype=torch.long)
        
        # Return input tensors separately for text and label
        return {
            'input_ids': text_inputs['input_ids'].squeeze(),
            'attention_mask': text_inputs['attention_mask'].squeeze(),
            'labels': text_inputs['input_ids'].squeeze(),  # The labels for causal language modeling are the input_ids
            'label_ids': label_id  # Label ID to pass to label embedding
        }


def theMain(
    # High-level
    vlads_device: bool = False,
    num_augments_per_file: int = 1, 
    # Paths
    train_data_path: Path = Path("data"),
    save_model_path: Path = Path("trained_model"),
    save_tokenizer_path: Path = Path("trained_tokenizer"),
):
    # Load training data (each element = (string of whole file, class label))
    train_lines = load_ldr_data(train_data_path / 'train', num_augments_per_file, is_eval_set=False)
    eval_lines = load_ldr_data(train_data_path / 'test', num_augments_per_file, is_eval_set=True)

    # Extract only the text part for tokenizer training
    train_texts = [text for text, label in train_lines]
    
    # Load & train tokenizer
    m3_tokenizer = load_tokenizer(int(52000), train_texts, save_tokenizer_path)
    printTokenStream(train_texts[0], m3_tokenizer)  # Example tokenization of the first training sample

    # Create a label map (Mapping from label names to unique IDs)
    unique_labels = list(set(label for _, label in train_lines))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Load custom LabelConditionedGPT2 model
    print("--- LOADING LabelConditionedGPT2 and Data Collator---")
    config = GPT2Config(
        vocab_size=m3_tokenizer.vocab_size,
        n_positions=m3_tokenizer.model_max_length,
    )
    model = LabelConditionedGPT2(config, num_labels=len(label_map))  # Pass the number of labels

    # Using standard data collator (optionally, you can define a custom data collator)
    data_collator = DataCollatorForLanguageModeling(tokenizer=m3_tokenizer, mlm=False)
    print("Finished loading")

    # Set training variables and train the model
    print("\n--- Initializing Training Arguments ---")
    training_args = TrainingArguments(
        output_dir=save_model_path,
        fp16=vlads_device,
        num_train_epochs=5,
        learning_rate=3e-6, 
        eval_steps=5000,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=1000,
        save_steps=10000, 
        save_total_limit=1,
        push_to_hub=False,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        overwrite_output_dir=True,
        metric_for_best_model="loss",   
        weight_decay=0.01,
        greater_is_better=False,
        lr_scheduler_type="cosine",     
    )

    # Saving training arguments so we can see what we trained with
    training_args_file = save_model_path / "train_args.txt"
    if training_args_file.exists(): training_args_file.unlink()
    with open(training_args_file, 'w') as f:
        for arg, value in vars(training_args).items():
            f.write(f"{arg}: {value}\n")
    print(f"Saved training arguments to {training_args_file}")

    # Tokenize data & put it in tensor format
    print("\n--- CONVERTING DATA TO TENSORS ---")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}   ~9 minutes when augments=1")
    train_dataset = LDRTextDataset(train_lines, m3_tokenizer, label_map)
    print("Completed training dataset")
    eval_dataset = LDRTextDataset(eval_lines, m3_tokenizer, label_map)
    print("Completed evaluation dataset")

    print("\n--- BEGINNING TRAINING ---")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # Start training
    trainer.train()

    print("\n--- TRANSFORMER TRAINED ---")
    model.save_pretrained(Path(save_model_path, "final"))
    print("\n--- TRAINED MODEL SAVED ---")

# Running the main function
import traceback
def main():
    try: 
        theMain()
    except Exception as e: 
        traceback.print_exception(type(e), e, e.__traceback__, limit=0)

if __name__ == "__main__": 
    typer.run(main)
