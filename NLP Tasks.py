'''
1. Data Preprocessing: Tokenisation 
'''
import re
text = "hello, world. is this--a school?"
result = re.split(r'([],.:;?_"()]| -- | \\s)', text)
result = [item.strip() for item in result if item.strip()]  #without strip we would also have leading and trailing whitespaces.
print(result)

''' 
2. Tokens to Token ID's

We now arrange them alphabetically, We need to build a vocabulary first to map the pregenerated tokens to any id's, aka unique integers 
'''

all_words = sorted(result) #sorting them alphabetically 
vocab_size = len(all_words)
print(vocab_size)

#Creating Vocabulary 

vocab = {token: integer for integer, token in enumerate(all_words)} #dictionary, adding index to the key value 
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break 

#We need to also map encoding and decoding to convert the token ids back to text 
class SimpleTokenizerV1:
    def __init__(self, vocab): 
        self.str_to_int = vocab #storing the vocab as class attribute for access in the encode and decode methods
        self.int_to_str = {i: s for s, i in vocab.item()}  #created an inverse vocabulary that maps the token id's back to the orignal text

    def encode(self, text):
        result = re.split(r'([],.:;?_"()]| -- | \\s)', text)
        result = [item.strip() for item in result if item.strip()]
        ids = [self.str_to_int[s] for s in result]
        return ids
    
    '''   def decode(self, ids):         
        text = " ".join([self.int_to_str[i] for i in ids]) 

        #text = re.sub(r'\s+([,.?!"()\\'])', r'\\1', text)   
        return text 
        '''

#We can enhace our vocbulary by adding <|unk|> token or <|endoftext|> (which acts as all the concatenated independent etxt sources) which clealry defines that these are not the part of the original text


text1 = "sneha"
text2 = "Maurya"
res = "<|endoftext|>".join((text1, text2))
print(res)



'''
3. Byte Pair Encoding - "tiktoken" lib in python helps to convert into smaller tasks: 
from importlib.metadata import version
import tiktoken
BPE is actually a compresion algorithm used in NLP to create subwords tokenisation. treating each charcter as a seperate token. 2)then look at pairs that actually match together and appear together (count pair frequencies) 3) Merge the most frequent pair that appear together. specially used for handling rare words.

'''

'''
4. Data Sampling with sliding windows - LLM is to generate the input–target pairs required for training an LLM
'''
'''context_size = 4         #1
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)


for i in range(1, context_size+1):     #input target pair creation
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
'''

    '''
    5. creating token embeddings - initialising weights with random values at first + encode positional information
    '''

    '''
    6. Attention Mechanisms - Mechanism in neural newtwork enabling the model to focus on specfic parts of the input sequence while making prediction, improving performance in tasks like machine translations tec.
    i) Simplified self-attention, ii) Self-attention, iii)Casual Attention, iv)multihead attention. word to word translation of any word in different language, won't form the same sentence grammatically correct. RNN's work really well for such kinds of jobs,
    RNN works fine with translating short sentences, but not with longer text as they do not have the direct access to previous words  in the input. 
    Basically Self - attention works like, This word is important for understanding that other word!” It focuses on how words in the same story relate to each other.
    We have attention weights(attention weight vector) z(i), x(i) - COntext aware is the summary for itself, evaluating the importance for itself. 
    '''

import torch 
inputs = torch.tensor(
    [[0.24, 0.56, 0.65]
     [0.45, 0.96, 0.67]
     [0.67, 0.67, 0.78]]  # we will take dot products with each form one another. to get the context vector, which serves as the attention weights
)

query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)    
