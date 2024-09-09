from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, GenerationConfig, AutoTokenizer
import typer
from scipy.spatial.transform import Rotation
from inference.inference import decode_v1, decode_v3

def valid_ldr(text):
    ldr = ""
    lines = text.split("\n")

    for line in lines:
        entries = line.split(" ")
        # Class token
        if entries[0] == '': continue

        if entries[0][0] == "<":
            ldr += '0 ' + entries[0] + "\n"
            continue

        # Invalid num entries
        if len(entries) != 11: 
            print(f"invalid line: {line}")
            continue
        
        # If valid line
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

def main(
    vlads_machine: bool = False,
    trained_tokenizer_dir: Path = Path("v1-1/M3_TOK_v1-1/trained_tokenizer"),
    trained_model_dir: Path = Path("v1-1/M3_GPT2_v1-1"),
):
    
    # Loading tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(trained_tokenizer_dir)
    print(f"\nTokenizer Loaded: {trained_tokenizer_dir.name}\n")
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not vlads_machine and torch.backends.mps.is_available(): torch.device("mps")
    model = AutoModelForCausalLM.from_pretrained(trained_model_dir).to(device)
    print(f"Model Loaded: {trained_model_dir.name}\n")

    # Prompt and config
    text_prompt = """<|VE|>"""
    prompt = torch.tensor([tokenizer.encode(text_prompt)], dtype=torch.long).to(device)
    generation_config = GenerationConfig(
        max_length=model.config.n_positions,
        do_sample=True,
        top_k=51,
        top_p=0.85,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    print("GENERATING OUTPUT...\n")
    
    # Generate tokens one by one
    output_tokens = prompt.clone().detach().cpu().numpy().tolist()[0]
    for i in range(1, 100):
        # Generate the next token
        output = model.generate(torch.tensor([output_tokens]).to(device), generation_config=generation_config, max_new_tokens=1)
        new_token = output[0, -1].item()
        output_tokens.append(new_token)
        decoded_token = tokenizer.decode(new_token, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        
        # print(f"Generated token: {decoded_token}")
        
        if new_token == tokenizer.eos_token_id or decoded_token == "":  # Stop generation if EOS token is generated
            break

    # PRINTING OUTPUT 
    print("\nTokens:")
    for token_id in output_tokens:
        decoded_token = tokenizer.decode(token_id, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded_token[0] == "<": 
            print(f"{decoded_token}\n")
        elif decoded_token == "\n": 
            print("\\n")
        else: 
            print(f"{decoded_token} |", end="")   

    formatted_output = decode_v1(output_tokens, tokenizer)

    # Converting to valid LDR
    print("\nFINAL LDR:")
    ldr = valid_ldr(formatted_output)

    # Saving ldr in .ldr file
    with open("output.ldr", "w") as f:
        f.write(ldr)
    
if __name__ == "__main__":
    typer.run(main)
