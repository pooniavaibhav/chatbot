"""

"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)


    def forward(self, input_seq, input_length, hidden=None):
        embeded = self.embedding(input_seq)
        packed = torch.nn.utils.rnn.pad_packed_sequence(embeded, input_length)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        return outputs, hidden

class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size

    def dot_score(self, hidden, encoder_output, dim=2):
        return torch.sum(hidden * encoder_output, dim=2)

    def forward(self, hidden, encoder_outputs):
        attn_energies = self.dot_score(hidden, encoder_outputs)
        attn_energies = attn_energies.t()
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongattnRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

    #Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_output):
          embeded = self.embedding(input_step)
          embeded = self.embedding_dropout(embeded)
          rnn_output, hidden = self.gru(embeded, last_hidden)
          attn_weights =  self.attn(rnn_output, encoder_output)
          context = attn_weights.bmm(encoder_output.transpose(0, 1))
          rnn_output = rnn_output.squeeze(0)
          context = context.squeeze(1)
          concat_input = torch.cat((rnn_output,context),1)
          concat_output = torch.tanh(self.concat(concat_input))
          output = self.out(concat_output)
          output = self.softmax(output, dim=1)
          return output, hidden
