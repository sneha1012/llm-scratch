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

