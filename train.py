import numpy as np

def initialized_feedforward_parameters(model):
    phi = 0
    return phi

def initialized_equilibrium_parameters(model):
    theta = 0
    return theta

def train(hyperparams: dict):
    
    while(True):
        for t in hyperparams("t-"):
            pass # neg phase
        
        for t in hyperparams("t+"):
            pass #pos phase
        
        pass

if __name__=="__main__":
    hyperparams = {
        "epsilon": 0.01,
        "eta": 0.1,
        "t-": 1,
        "t+":1
    }
