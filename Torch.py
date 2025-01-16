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


#Multilayer perceptron in PyTroch

class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):    
        super().__init__()

        self.layers = torch.nn.Sequential(

            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),    #the linear layer takes on input features and returns nodes a arguments
            torch.nn.ReLU(),               

            # 2nd hidden layer
            torch.nn.Linear(30, 20),    
            torch.nn.ReLU(), #non linear activation fucntions are placed between hidden layers

            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        logits = self.layers(x) #outputs of the last layers called logits
        return logits    
model = NeuralNetwork(50, 3)


num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)       

torch.manual_seed(123)   
X = torch.rand((1, 50))
out = model(X)
print(out)

#Cerating a Toy dataset 
from torch.utils.data import Dataset

class ToyDataset(Dataset):
    def __init__(self, X, y):
        self.features = X
        self.labels = y 

    def __getitem__(self, index):    #particular index from the data (features inputs)     
        one_x = self.features[index]     
        one_y = self.labels[index]       
        return one_x, one_y             

    def __len__(self):
        return self.labels.shape[0]      

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

from torch.utils.data import DataLoader
torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,     
    batch_size=2,
    shuffle=True,          
    num_workers=0     
)


##DataLoaders Class, rtakes care of how data is shuffled and batched
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=2,
    shuffle=True,
    num_workers=0,  #data loadeing will be done i main process and not in seperate work proesses, paralle with reprocessing.
    drop_last=True
)


##Neural Network training in Pytorch using the toy dataset we created earlier
import torch.nn.functional as F
torch.manual_seed(123) #code reproducible and debugging is easier for any randomly generated stuff in the program. 
model = NeuralNetwork(num_inpouts = 2, num_outputs = 2)
optimizer = torch.optim.SGD(
    model.parameters() #SGD tweaks the model parameter towards the negative grdaient thus minimisng th loss , model parameter retrives all part of program that neeeds to be updated during traning weights and biases.
)
