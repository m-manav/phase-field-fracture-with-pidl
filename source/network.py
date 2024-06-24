import torch
import torch.nn as nn
import numpy as np
import warnings

class SteepTanh(nn.Module):
    def __init__(self, coeff):
        super(SteepTanh, self).__init__()
        self.coeff = coeff
        
    def forward(self, x):
        activation = nn.Tanh()
        return activation(self.coeff*x)
    
class SteepReLU(nn.Module):
    def __init__(self, coeff):
        super(SteepReLU, self).__init__()
        self.coeff = coeff
        
    def forward(self, x):
        activation = nn.ReLU()
        return activation(self.coeff*x)
    
class TrainableTanh(nn.Module):
    def __init__(self, init_coeff):
        super(TrainableTanh, self).__init__()
        self.coeff = nn.Parameter(torch.tensor(init_coeff))
    def forward(self, x):
        activation = nn.Tanh()
        return activation(self.coeff*x)
    
class TrainableReLU(nn.Module):
    def __init__(self, init_coeff):
        super(TrainableReLU, self).__init__()
        self.coeff = nn.Parameter(torch.tensor(init_coeff))
    def forward(self, x):
        activation = nn.ReLU()
        return activation(self.coeff*x)
    
    

class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, activation, init_coeff=1.0):
        super(NeuralNet, self).__init__()
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.neurons = neurons
        self.n_hidden_layers = n_hidden_layers

        # Activation function
        self.name_activation = activation
        self.init_coeff = init_coeff
        self.trainable_activation = False

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)
        self.activations, self.trainable_activation = activations(activation, init_coeff, n_hidden_layers)

    def forward(self, x):
        if self.trainable_activation:
            x = self.activations[0](self.input_layer(x))
            for j, l in enumerate(self.hidden_layers):
                x = self.activations[j+1](l(x))
            return self.output_layer(x)
        else:
            x = self.activations(self.input_layer(x))
            for j, l in enumerate(self.hidden_layers):
                x = self.activations(l(x))
            return self.output_layer(x)
        
    
def activations(activation, init_coeff, n_hidden_layers=1):
    if activation == 'SteepTanh':
        activations = SteepTanh(init_coeff)
        trainable_activation = False
    elif activation == 'SteepReLU':
        activations = SteepReLU(init_coeff)
        trainable_activation = False
    elif activation == 'TrainableTanh':
        activations = nn.ModuleList([TrainableTanh(init_coeff) for _ in range(n_hidden_layers)])
        trainable_activation = True
    elif activation == 'TrainableReLU':
        activations = nn.ModuleList([TrainableReLU(init_coeff) for _ in range(n_hidden_layers)])
        trainable_activation = True
    else:
        warnings.warn('Prescribed activation does not match the available choices. The default activation Tanh is in use.')
        activations = nn.Tanh()
        trainable_activation = False

    return activations, trainable_activation


def init_xavier(model):
    activation = model.name_activation
    init_coeff = model.init_coeff
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            if activation == 'TrainableReLU' or activation == 'SteepReLU':
                g = nn.init.calculate_gain('leaky_relu', np.sqrt(init_coeff**2-1.0))
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

            if activation == 'TrainableTanh' or activation == 'SteepTanh':
                g = nn.init.calculate_gain('tanh')/init_coeff
                torch.nn.init.xavier_uniform_(m.weight, gain=g)
                # torch.nn.init.xavier_normal_(m.weight, gain=g)
                m.bias.data.fill_(0)

    model.apply(init_weights)

