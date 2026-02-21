import numpy as np

def market_model(role):

    demand = np.random.randint(500,2000)
    supply = np.random.randint(2000,8000)

    market_tightness = demand / supply

    return demand, supply, market_tightness
