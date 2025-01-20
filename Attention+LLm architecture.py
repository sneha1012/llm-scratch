#Computing the attention weights step by step, we will focus on three matrices (weight matrices) w(q,k,v). Query, key, value 
import torch
import torch.nn as nn
class SelfAttention_v1(nn.Module):  #class is derived from nn.Module  from torch functions 
    def __init__(self, d_in, d_out): #initialises the weights and everything to be used 
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
        self.W_value  = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)   #If we were to use the weight metrices for model training, we would set this up to grad= true 
    
    def forward(self, x):


        #vector calculation 
        queries =  x @ self.W_query
        keys= x @ self.W_key
        values = x  @self.W_value

        #attention score  
        attn_scores = queries @ keys.T  #normalising it in the next step. 
        attn_weights = torch.softmax( 
            attn_scores / keys.shape[-1] **0.5, dim =-1
        )
        context_vec = attn_weights @ values
        return context_vec 
    
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)   
print(sa_v1(inputs)) 

##Applying a casual attention mask 
'''
1. Attention scores - apply softmax to normalise 
2. attention weights - mask with 0's above diagonal 
3. masked attention scores - normalise (row amounts =1 )
'''

queries = sa_v2.W_query(inputs)
keys = sa_v2.W_keys(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim= -1)
print(attn_weights)


#values above the diagonal are zer0 

context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))  #tril returns the lower half of the 2d matrix, setting all other values to 0
print(mask_simple)

#zeroing  out values above the triangle.
masked_simple = attn_weights*mask_simple

