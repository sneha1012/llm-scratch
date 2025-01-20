    #vector calculation 
        queries =  x @ self.W_query
        key= x @ self.W_key
        value = x  @self.W_value

        #attention score 
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1] **0.5, dim =-1
        )
        context_vec = attn_weights @ values
        return context_vec