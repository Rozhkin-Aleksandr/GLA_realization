# Gated Linear Attention Transformers with Hardware-Efficient Training

## Overview

This project implements the ideas proposed in the paper **"Gated Linear Attention Transformers with Hardware-Efficient Training."** The goal is to enhance the efficiency of transformer models, specifically by modifying the basic GPT-2 architecture to incorporate gated linear attention mechanisms.

## Repository Structure

The repository is organized as follows:

├── docs/               # Documentation files  
├── models/             # Pre-trained models and model definitions  
├── requirements.txt    # Python dependencies  
├── data/               # Dataset files  
├── draft.ipynb         # Jupyter notebook for experimentation  
├── notebooks/          # Additional Jupyter notebooks  
└── src/                # Source code  
    ├── main.py         # Main training script   
    ├── gla_model.py    # Implementation of the Gated Linear Attention model  


## Installation

To set up the project, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Rozhkin-Aleksandr/GLA_realization.git
   cd GLA_realization
Create a virtual environment (optional but recommended):

  ```bash
  python -m venv venv
  source venv/bin/activate # On Windows use venv\Scripts\activate
```
Install the required dependencies:


```bash
pip install -r requirements.txt
```
## Usage
Training the Model
To train the model, you can run the main.py script. This script contains all the necessary configurations and parameters for training the GLA transformer model.

```bash
python src/main.py
```
## Modifying the Model
The model architecture is defined in gla_model.py. You can modify the architecture, hyperparameters, and other settings in this file according to your requirements.

## Experimentation
For experimentation and analysis, refer to the draft.ipynb and notebooks/ directory. These Jupyter notebooks contain various experiments conducted with the model and can serve as a guide for your own experiments.

## Contact
For any inquiries or feedback, please contact:

Name: Aleksandr Rozhkin  
Email: aerozhkin@edu.hse.ru
