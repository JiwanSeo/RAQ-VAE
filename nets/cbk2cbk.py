import torch
from torch import nn
import random


# Encoder
class CdBkEncoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.dropout(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell
        
        
# decoder
class CdBkDecoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, cell):
        embedded = self.dropout(x)  # [1, batch_size, emd dim]
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        out = output.squeeze(0)
        return out, hidden, cell
    
    
# Seq2Seq
class CdBk2CdBk(nn.Module):
    def __init__(self, encoder, decoder, embed_layer_enc, embed_layer_dec, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.embed_layer_enc = embed_layer_enc
        self.embed_layer_dec = embed_layer_dec
        
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, 'Hidden dimensions of encoder decoder must be equal'
        assert encoder.n_layers == decoder.n_layers, 'Encoder and decoder must have equal number of layers'

    
    def embed_src(self, cd):
        return self.embed_layer_enc(cd.unsqueeze(0))
    
    def embed_trg(self, cd):
        return self.embed_layer_dec(cd.unsqueeze(0))
    
    def forward(self, src, trg):
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]

        outputs = torch.zeros(trg_len, batch_size, 64).to(self.device)

        # initial hidden state
        hidden, cell = self.encoder(self.embed_layer_enc(src))
        
        input = self.embed_trg(trg[0])
        
        # cross forcing
        teacher_forcing_ratio = 0.5
        for t in range(0, trg_len): 
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[t] = output
        
            if (t % 2 == 0) & (t < src.shape[0]*2):
                input = self.embed_src(trg[t//2])
            else:
                input = self.embed_trg(trg[t])
             
        return outputs