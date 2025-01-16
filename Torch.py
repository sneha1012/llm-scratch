import torch
tensor0d = torch.tensor(1) #we can create objects
tensor1d = torch.tensor([1,2,3])
tensor2d = torch.tensor([[1,2],
                        [3,4]])
#printing the shape like in python 
print(tensor2d.shape)  
print(tensor2d.reshape(4,1))    
print(tensor2d.view(4,1))     #similar to numpy conversions      

#Transposing - flipping is accorss it's diagonal (left to right)
print(tensor2d.T)

#Multiply two Matrices 
print(tensor2d.matmul(tensor2d.T)) #or just use @ like below:
print(tensor2d @ tensor2d.T)

#PyTorch adopts the dfault 64-bit integer data type   
tensor1d = torch.tensor([1,2,3])     
print(tensor1d.dtype)  



#logisitic Regression forward pass
import torch.nn.functional as F
from torch.autograd import grad
y = torch.tensor([1.0]) #true label
x1 = torch.tensor([1.1])   #input features 
w1 = torch.tensor([2.2], requires_grad= True) 
b = torch.tensor([0.0], requires_grad = True)
z = x1 * w1 + b  

a = torch.sigmoid(z) #activation 
loss = F.binary_cross_entropy(a, y ) #cross 
loss.backward()
print(w1.grad)
print(b.grad)
