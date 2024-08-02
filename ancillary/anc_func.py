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