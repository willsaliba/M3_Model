import os
os.system('clear')

from tokenizers import Tokenizer, pre_tokenizers, Regex, normalizers, decoders
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import Replace
from tokenizers.pre_tokenizers import WhitespaceSplit, Split
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from transformers import PreTrainedTokenizerFast

### HYPER PARAMS ###
vocab_size = 52000
max_context_window = 2048
training_data_path = "data/original_datasets/omr8/train"
trained_tokenizer_path = "trained_m3"

#TOKENIZER MODEL - BPE algorithm for inc compression & interpretability of unseen characters
m3 = Tokenizer(WordLevel(unk_token="<UNK>")) 

#NORMALISATION
m3.normalizer = normalizers.Sequence([
    Replace(Regex(r'^.*?\K\s'), " Colour "), 
    Replace(Regex(r'^(?:[^\s]*\s){2}[^\s]*\K\s'), " Position "),
    Replace(Regex(r'^(?:[^\s]*\s){6}[^\s]*\K\s'), " Rotation "),
    Replace(Regex(r'^(?:[^\s]*\s){16}[^\s]*\K\s'), " Shape "), 
    Replace(Regex(r'(?<=\.\d{3})\d+'), ""), #removes the last 3 decimal places
    Replace(Regex(r'^[^1].*$'), ""), #removes non-brick lines
    Replace('.000', ""),
])

#PRETOKENIZATION 
m3.pre_tokenizer = pre_tokenizers.Sequence([
    WhitespaceSplit(),
    Split(pattern=Regex(r"-|\.\d{3}"), behavior="isolated"),
])

#TRAINING
print(f"\n--- LOADING TRAIN DATA ---\nsource: {training_data_path}\n")
files = []
for file_name in os.listdir(training_data_path):
    file_path = os.path.join(training_data_path, file_name)
    files.append(file_path)

print("\n--- TRAINING TOKENIZER ---\n")
m3_trainer = WordLevelTrainer(
    vocab_size = vocab_size,
    show_progress=True,
    special_tokens=["[UNK]", "[SOS]", "[EOS]", "[PAD]"] #unk, start of sequence, end of sequence, padding
)
m3.train(files, m3_trainer)

print(f"\n\n--- SAVING TOKENIZER ---\nlocation: {trained_tokenizer_path}\n")
#removing pre-exisiting files in tokenizer_save_path
for file_name in os.listdir(trained_tokenizer_path):
    file_path = os.path.join(trained_tokenizer_path, file_name)
    os.remove(file_path)
#converting tokenizer to transformer tokenizer and saving
M3 = PreTrainedTokenizerFast(tokenizer_object=m3)
M3.model_max_length = max_model_length
M3.save_pretrained(trained_tokenizer_path)

print("\n--- M3 SAVED ---\n")
print(f"FINAL SIZE: {len(m3.get_vocab())}\n")

def post_processing(text):
    result = ""
    i = 0
    while i < len(text):
        ltr = text[i]
        #if label
        if ltr == 'C': 
            i += 7
            continue
        elif ltr == 'P':
            i += 9
            continue
        elif ltr == 'R':
            i += 9
            continue
        elif ltr == 'S':
            i += 6
            continue

        #if negative sign
        if ltr == '-':
            result += ltr
            i += 2
            continue
        
        result += ltr
        i += 1

    return result

""" ---------------------------------------------- """

test_text = """0 FILE 7181 - TIE Interceptor - UCS__6_8_13.mpd
0 Main
0 Name: 7181 - TIE Interceptor - UCS__6_8_13.mpd
0 Author: LTRON
1 0 -308.464465 -279.078750 84.596443 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 4282.dat
1 0 -246.239049 -235.783515 51.192078 -0.707107 0.707107 0.000000 -0.678823 -0.678823 0.279000 0.197283 0.197283 0.960000 4151.dat
1 1 -321.192391 -258.116396 161.791009 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 3460.dat
1 1 -321.192391 -286.016396 65.791009 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 2412.dat
1 0 -309.878679 -280.137559 126.547534 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 3460.dat
1 0 -295.736539 -252.611105 170.601877 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 3623.dat
1 0 -232.096909 -188.727060 162.446421 0.000000 0.707107 -0.707107 -0.279000 -0.678823 -0.678823 -0.960000 0.197283 0.197283 3934.dat
1 0 -251.895905 -208.032933 126.413815 -0.707107 0.707107 0.000000 -0.678823 -0.678823 0.279000 0.197283 0.197283 0.960000 3795.dat"""

M3_encoding = M3.encode(test_text)
print(M3_encoding)

M3_decoding = M3.decode(M3_encoding)
decoding = post_processing(M3_decoding)
for i in range(len(decoding)):
    token = decoding[i]
    print(token, end="")
    if i < 5: continue
    if token == 't' and decoding[i-1] == 'a' and decoding[i-2] == 'd': print("\n", end="") 







"""inspecting pretokenization pattern"""

# normalised_text = m3.normalizer.normalize_str(test_text)
# pretokenized = m3.pre_tokenizer.pre_tokenize_str(normalised_text)
# for tup in pretokenized:
#     token = tup[0]
#     print(f"[{token}] ", end="")
#     if token.endswith(".dat"): print("\n")

# gpt4_pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
# m2_pre = pre_tokenizers.Split(pattern=Regex(gpt4_pattern), behavior="isolated")
# pretokenized = m2_pre.pre_tokenize_str(normalised_text)
# for tup in pretokenized:
#     token = tup[0]
#     print(f"[{token}] ", end="")
#     if token.endswith(".dat"): print("\n")


"""inference tokens"""
# tokenizer.add_special_tokens({'sos_token': '[SOS]'}) #start of sequence
# tokenizer.add_special_tokens({'eos_token': '[EOS]'}) #end of sequence
# tokenizer.add_special_tokens({'pad_token': '[PAD]'}) #padding for transformer

"""Examples"""
# test_text = """0 FILE 7181 - TIE Interceptor - UCS__6_8_13.mpd
# 0 Main
# 0 Name: 7181 - TIE Interceptor - UCS__6_8_13.mpd
# 0 Author: LTRON
# 1 0 -308.464465 -279.078750 84.596443 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 4282.dat
# 1 0 -246.239049 -235.783515 51.192078 -0.707107 0.707107 0.000000 -0.678823 -0.678823 0.279000 0.197283 0.197283 0.960000 4151.dat
# 1 1 -321.192391 -258.116396 161.791009 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 3460.dat
# 1 1 -321.192391 -286.016396 65.791009 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 2412.dat
# 1 0 -309.878679 -280.137559 126.547534 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 3460.dat
# 1 0 -295.736539 -252.611105 170.601877 0.000000 0.707107 0.707107 0.279000 -0.678823 0.678823 0.960000 0.197283 -0.197283 3623.dat
# 1 0 -232.096909 -188.727060 162.446421 0.000000 0.707107 -0.707107 -0.279000 -0.678823 -0.678823 -0.960000 0.197283 0.197283 3934.dat
# 1 0 -251.895905 -208.032933 126.413815 -0.707107 0.707107 0.000000 -0.678823 -0.678823 0.279000 0.197283 0.197283 0.960000 3795.dat"""