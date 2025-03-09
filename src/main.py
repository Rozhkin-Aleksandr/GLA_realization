from gla_model import GLA
from train_loop import train_gla
from prepare_data import load_and_prepare_data

if __name__ == '__main__':
    train_loader = load_and_prepare_data()
    model = GLA()
    train_gla(model, train_loader)
    
    