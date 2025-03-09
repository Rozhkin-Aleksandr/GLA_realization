from prepare_data import prepare_data
from gla_model import GLA
from train_loop import train_gla

if __name__ == '__main__':
    train_loader = load_and_prepare_data()
    model = GLA()
    train_gla(model, train_loader)
    
    