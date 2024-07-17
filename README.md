Hi Zach :)

Everything should be as you expect. The only issue is if you want to try training 
locally and you have the latest version of transformers it seems to break. To fix
see the following isntructions.

Also for now there is only a train file but I plan to make an inference file soon 
which will automatically load the M3 tokenizer and trained model, and also contain
a custom decode function which ensures output is valid LDR. 
(some clean up stuff is needed like removing special tokens and adding newline characters)

---IF TRAINING ON MAC---  
-> GPU: need to change source code in transformers library  
see MPA's post: https://discuss.huggingface.co/t/runtimeerror-placeholder-storage-has-not-been-allocated-on-mps-device/42999/18

-> CPU (quick solution but slow training), just add following line to training_args
no_cuda=True,