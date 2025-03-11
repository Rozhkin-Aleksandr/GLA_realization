# Gated Linear Attention Transformers with Hardware-Efficient Training

## Overview

The primary objective of this research is to improve the efficiency of transformer models by introducing gated linear attention mechanisms into a modified architecture. While the foundational structure is loosely inspired by GPT-2, the implementation diverges significantly from the original model. Specifically, the only components retained from GPT-2 are the tokenizer and embedding layers, as well as the layer normalization modules. The attention mechanisms and multi-layer perceptron (MLP) layers have been entirely redesigned to incorporate gated linear attention, aiming to reduce computational complexity while maintaining or enhancing performance.

For this study, the 'wikitext-2-raw-v1' dataset was selected due to its manageable size of approximately 10 million tokens, making it suitable for experimentation on a CPU-based environment, such as a personal laptop. This choice ensures computational feasibility while allowing for thorough evaluation of the proposed architectural modifications. The focus on reengineering the attention and MLP layers underscores the novelty of this approach, as it seeks to explore the potential of gated linear attention in optimizing transformer efficiency without relying on the core components of the original GPT-2 architecture.

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
2. **Create a virtual environment (optional but recommended)**:  

  ```bash
  python -m venv venv
  source venv/bin/activate # On Windows use venv\Scripts\activate
```
3. **Install the required dependencies**:  


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
The model architecture is defined in gla_model.py. You can modify the architecture, hyperparameters in config.py, and other settings in this file according to your requirements.

## Experimentation
For experimentation and analysis, refer to the draft.ipynb and notebooks/ directory.

## Contact
For any inquiries or feedback, please contact:

Name: Aleksandr Rozhkin  
Email: aerozhkin@edu.hse.ru
