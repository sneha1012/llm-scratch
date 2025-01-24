#Computing the attention weights step by step, we will focus on three matrices (weight matrices) w(q,k,v). Query, key, value 


##Applying a casual attention mask 
'''
1. Attention scores - apply softmax to normalise 
2. attention weights - mask with 0's above diagonal 
3. masked attention scores - normalise (row amounts =1 )
'''
import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):  
    def __init__(self, d_in, d_out): 
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
        self.W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
    
    def forward(self, x):
        queries = x @ self.W_query
        keys = x @ self.W_key
        values = x @ self.W_value

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec 

# Define input parameters
d_in = 8
d_out = 8
inputs = torch.rand(4, d_in)

# Instantiate and test the SelfAttention class
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))

# Applying a casual attention mask
queries = inputs @ sa_v1.W_query
keys = inputs @ sa_v1.W_key
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

print("Attention Weights (Before Mask):")
print(attn_weights)

#The mask ensures that tokens can only attend to themselves and previous tokens in the sequence. This is crucial in causal attention for tasks like language modeling, where a token shouldn’t depend on future tokens.
# Create a lower triangular mask
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("Mask:")
print(mask_simple)            #causal Attention

# Zeroing out values above the diagonal
masked_simple = attn_weights * mask_simple
print("Masked Attention Weights:")
print(masked_simple)


#Multi head - aalows splitting the attention mechanism into multiple heads. each learns differetn aspects of the data, allowing models to simultaneously attend to information from different subspaces from different positions. 
#Causal Attention - we need to hide the future infor, prediction should pure;y depend on the previous words.

row_sums = masked_simple.sum(dim=-1, keepdim=True) #normalising the mask to sum up to 1 in each row. 
masked_simple_norm = masked_simple / row_sums
print(masked_simple_norm)

#Masking additional attention weights with dropouts helpful to reduce overfitting. 

