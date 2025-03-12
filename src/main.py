from gla_model import GLA
from train_loop import train_gla
from prepare_data import load_and_prepare_data
from evaluation import evaluate_perplexity
import torch


if __name__ == '__main__':
    train_loader, test_loader = load_and_prepare_data()
    model = GLA()
    train_gla(model, train_loader)
    model.load_state_dict(torch.load('best_model_weights.pth'))
    evaluate_perplexity(model, test_loader)
    
    