
import torch.nn as nn
import torch

class Encoder(nn.Module):
 
    def __init__(self,input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(input_dim), hidden_dim),
            nn.ReLU()
        )
        
        self.f_mu = nn.Linear(hidden_dim, output_dim)
        self.f_var = nn.Linear(hidden_dim, output_dim)

    def encode(self, img):
        h1 = self.model(img)
        mu = self.f_mu(h1)
        logvar = self.f_var(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )
        self.f_mu = nn.Linear(hidden_dim, output_dim)
        self.f_std = nn.Linear(hidden_dim, output_dim)
        
    def decode(self,z):
        h1 = self.model(z)
        return self.f_mu(h1), self.f_std(h1)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
        
    def forward(self, z):
        mu, logvar = self.decode(z)
        output = self.reparameterize(mu,logvar)
        return output  
