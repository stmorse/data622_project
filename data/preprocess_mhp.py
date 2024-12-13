import pickle
import time
import numpy as np
import pandas as pd

import transformers

DATA_PATH = 'data_ts/'
DATA_CAPS = {'train': 500, 'test': 50}

def main():
    tokenizer = transformers.AutoTokenizer.from_pretrained("google-bert/bert-base-cased")

    data_types = ['train', 'test']

    t0 = time.time()

    # load data
    print(f'Loading data ... {time.time()-t0:.3f}')
    raw = {}
    for t in data_types:
        raw[t] = pd.read_csv(f'{DATA_PATH}{t}.csv')
        raw[t] = raw[t].dropna()
        print(t, raw[t].shape)

    # store all tokens used in training and validation
    all_tokens = []

    # tokenize data
    tokenized = {}
    for t in data_types:
        print(f'Tokenize {t} data ... {time.time()-t0:.4f}')
        tokenized[t] = []
        for i, story in enumerate(raw[t]['text'][:DATA_CAPS[t]]):
            story_tokenized = []
            try:
                tokens = tokenizer(story)['input_ids']
                all_tokens.extend(tokens)
                for k, token in enumerate(tokens):
                    story_tokenized.append([
                        k,
                        token
                    ])
                tokenized[t].append(story_tokenized)
            except Exception as e:
                print(f'Exception at line {i}: {e}')

    # convert to set
    all_tokens = set(all_tokens)

    # conversion dicts
    token_dict = {token: k for k, token in enumerate(all_tokens)}
    token_reverse_dict = {v: k for k, v in token_dict.items()}

    # update training/validation data with new tokens
    print(f'Adjusting tokens ... {time.time()-t0:.4f}')
    for t in data_types:
        for story in tokenized[t]:
            for entry in story:
                entry[1] = token_dict[entry[1]]

    # save to disk
    print(f'Saving to file ... {time.time()-t0:.4f}')
    with open(f'{DATA_PATH}data_mhp.pkl', 'wb') as f:
        pickle.dump(tokenized, f)
    
    with open(f'{DATA_PATH}token_dicts_mhp.pkl', 'wb') as f:
        pickle.dump({
            'token_dict': token_dict, 
            'token_reverse_dict': token_reverse_dict
            }, f)
        
    print(f'Complete.  {time.time()-t0:.3f}')
    

if __name__=="__main__":
    main()
