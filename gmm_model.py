from folder_images_dataset import folder_datasets, FolderImages

import torch
from torch import nn, optim, distributions

# g(x) = [ 1/ (sigma * sqrt(2pi)) ] * exp ((x-mu)/2sigma**2) 

class GMMModel(nn.Module):
    
    def __init__(self, num_gaussians=5, input_dim=3):
        
        super().__init__()

        self.means = nn.Parameter(torch.ones(input_dim, num_gaussians))
        self.stds = nn.Parameter(torch.ones(num_gaussians, num_gaussians))
 
        self.weights = torch.ones((num_gaussians,), requires_grad=True)
        self.mu = torch.randn((num_gaussians,), requires_grad=True)
        self.sigma = torch.rand((num_gaussians,), requires_grad=True)
        
        parameters = [self.weights, self.mu, self.sigma]
        self.optimizer = optim.SGD(parameters, lr=0.01, momentum=0.9)
    
    def E_step(self, num_gaussians):
        
        self.num_gaussians = num_gaussians
        self.weights = torch.randn(num_gaussians)
        self.mu = torch.rand(num_gaussians)
        self.sigma = 

    @staticmethod
    def softmax(x):
        return np.exp(-x) / np.sum(np.exp(-x))

    def forward(self, x):
        
        print (self.weights.shape, self.mu.shape, self.sigma.shape)
        sampler = distributions.Categorical(self.weights)
        comp = distributions.Independent(distributions.Normal(self.mu, self.sigma), 1)
        self.gmm = distributions.MixtureSameFamily(sampler, comp)
        
        self.optimizer.zero_grad()
        loss = - self.gmm.log_prob(x).mean() 
        loss.backward()
        self.optimizer.step()

        print(loss)

        return self.gmm

if __name__ == "__main__":
    
    import numpy as np

    x = np.array([[0.1,0.2,0.3], [0.15,0.35,0.3], [0.01,0.5,0.4]])
    
    w = np.array([0.1,0.6,0.3]) 

    mu, sigma = np.mean(x), np.std(x)
    
    g = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-(x - mu).dot(w.transpose()) / (2 * sigma**2))
    
    print (g, GMMModel.softmax(g))
    exit()

    model = GMMModel()
    
    x = torch.arange(0, 1, 2e-4 * 0.33)[:15000].reshape((5000,3)) #this can be an arbitrary x samples

    for y in range(len(x)):
        print (model.forward(x[y:y+1].transpose(1,0)[:,0]).shape)        

        exit() 
