import numpy as np

class model:
    def __init__(self, weights):
        #Hyperparameters
        self.input_size = 3 # including bias
        self.hidden_size = 8
        self.output_size = 1
        self.W1, self.W2 = weights
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def forward_propagate(self, X):
        #X = np.concatenate((np.ones((X.shape[0], 1)), X), axis = 1) #For now
        Z1 = np.dot(self.W1, X.T) 
        A1 = self.relu(Z1) #relu for hidden
        A1 = np.concatenate((np.ones((1,A1.shape[1])), A1))
        Z2 = np.dot(self.W2, A1)
        A2 = self.sigmoid(Z2)#sigmoid for output
        return A2 
    
    def loss(self, X, y):
        # returns the total cross entropy loss on the whole dataset
        y_pred = self.forward_propagate(X)
        loss = y*np.log(y_pred) + (1 - y)*(np.log(1 - y_pred))
        loss = -np.sum(loss) / y_pred.shape[1]
        return loss
        
        
