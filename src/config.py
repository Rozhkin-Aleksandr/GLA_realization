max_lr = 3e-4
weight_decay = 0.01
gradient_clip = 1.0
initial_lr = 3e-5
final_lr = 3e-5
epochs = 1

warmup_tokens = 0.1e6 
total_tokens = 2.3e6    

vocab_size = 50256
c = 5
hidden_dim = 768

max_seq_len = 64
batch_size = 16
batch_size_tokens = batch_size*max_seq_len  
