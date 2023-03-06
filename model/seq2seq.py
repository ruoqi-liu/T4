import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from torch.autograd import Function
from torch.nn import Module
from torch import tensor

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat','concat2']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        elif self.method == 'concat2':
            self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def concat_score2(self, hidden, encoder_output):
        h = torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)
        h = torch.cat((h, hidden*encoder_output),2)
        energy = self.attn(h).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, mask):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'concat2':
            attn_energies = self.concat_score2(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        # attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        attn_energies = torch.exp(attn_energies)
        attn_energies = attn_energies * mask.float()
        attn_energies_sum = attn_energies.sum(dim=0)
        attn_energies = attn_energies / (attn_energies_sum.unsqueeze(0) + 0.000001)

        return attn_energies

class Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, x_static_size, emb_dim, hid_dim, n_layers, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim

        self.n_layers = n_layers

        self.embedding = nn.Linear(input_dim, emb_dim)  # no dropout as only one layer!

        self.embedding_static = nn.Linear(x_static_size, emb_dim)

        self.x_static_size = x_static_size

        self.rnn = nn.LSTM(emb_dim+1, hid_dim, n_layers, batch_first=True)

        self.fc_out = nn.Linear(hid_dim+emb_dim+1, output_dim)

        self.dropout = nn.Dropout(dropout)

        self.attention_encoder = torch.nn.Sequential(
            torch.nn.Linear(hid_dim, 1),
            torch.nn.Tanh(),
        )

        self.ps_out = torch.nn.Sequential(
            nn.Linear(hid_dim+emb_dim, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, 1, bias=False)
        )

        self.device = device

        self.lambd = 0

    def set_lambda(self, lambd):
        self.lambd = lambd

    def softmax_masked(self, inputs, mask, dim=0, epsilon=0.0000001):
        # inputs = mask = [src len, batch size]
        inputs_exp = torch.exp(inputs)
        inputs_exp = inputs_exp * mask.float()
        inputs_exp_sum = inputs_exp.sum(dim=dim)
        inputs_attention = inputs_exp / (inputs_exp_sum.unsqueeze(dim) + epsilon)

        return inputs_attention

    def forward(self, src, x_static, treatment):

        embedded = self.embedding(src)
        embedded = self.dropout(embedded)

        embedded_static = self.embedding_static(x_static)

        treatment_hist = treatment[:,:(src.shape[1] - 1)]
        treatment_zero = (torch.zeros(treatment_hist.shape[0])).unsqueeze(1).to(self.device)
        treatment_hist = torch.cat((treatment_zero, treatment_hist), dim=1)
        treatment_hist = treatment_hist.unsqueeze(-1)
        embedded = torch.cat((embedded, treatment_hist), dim=-1)

        outputs, (h,c) = self.rnn(embedded)  # no cell state!

        src_mask = (src.sum(dim=-1) != 0).long()
        att_enc = self.attention_encoder(outputs).squeeze(-1)

        att_normalized = self.softmax_masked(att_enc, src_mask)
        hidden = torch.sum(outputs * att_normalized.unsqueeze(-1), dim=1)
        treatment_cur = treatment[:, -1].unsqueeze(-1)

        hidden_con = torch.cat((hidden, embedded_static), dim=-1)

        ps_pred = self.ps_out(hidden_con).squeeze(-1)

        return h, c, outputs, hidden, ps_pred, src_mask


class AttentionDecoder(nn.Module):
    def __init__(self, output_dim, x_static_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(1, emb_dim)

        self.embedding_static = nn.Linear(x_static_size, emb_dim)

        self.x_static_size = x_static_size

        self.attn_f = Attn('general', hid_dim)

        self.rnn = nn.LSTM(emb_dim*2+1, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out_1 = torch.nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, 1))

        self.fc_out_0 = torch.nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, 1))

        self.ps_out = torch.nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hid_dim, 1))

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, x_static, treatment_cur, treatment_next, hidden, cell, encoder_outputs, mask):

        input = input.unsqueeze(-1)
        embedded = self.embedding(input)

        # print(x_static.shape)
        embedded_static = self.embedding_static(x_static)

        embedded = embedded.unsqueeze(0) if len(embedded.shape) != 2 else embedded

        embedded = torch.cat((embedded, embedded_static), dim=-1)


        treatment_cur = treatment_cur.unsqueeze(-1)

        emb_con = torch.cat((embedded, treatment_cur), dim=-1)

        emb_con = self.dropout(emb_con)

        output, (hidden,cell)= self.rnn(emb_con.unsqueeze(1), (hidden, cell))

        # attention layer
        attn_weights = self.attn_f.forward(output, encoder_outputs, mask)
        context = torch.sum(encoder_outputs * attn_weights.unsqueeze(-1), dim=1)

        output = torch.cat((output.squeeze(1), context), dim=-1)

        ps_output = self.ps_out(output).squeeze(-1)

        treatment_next = treatment_next.unsqueeze(-1)
        prediction = treatment_next * self.fc_out_1(output) + (1-treatment_next) * self.fc_out_0(output)
        prediction_cf = treatment_next * self.fc_out_0(output) + (1 - treatment_next) * self.fc_out_1(output)

        return (prediction, prediction_cf), (hidden, cell), output, ps_output, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, x_static, treatment, teacher_forcing_ratio=0.5):

        batch_size = trg.shape[0]
        src_len, trg_len = src.shape[1], trg.shape[1]-1
        trg_vocab_size = self.decoder.output_dim

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        outputs_cf = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        decoder_attentions = torch.zeros(batch_size, trg_len, src_len).to(self.device)
        ps_outputs = torch.zeros(batch_size, trg_len+1).to(self.device)

        # last hidden state of the encoder is the context
        hidden, cell, encoder_outputs, patient_representations,  ps_pred, src_mask = self.encoder(src, x_static, treatment[:, :src_len])
        ps_outputs[:, 0] = ps_pred

        input = trg[:, 0].squeeze(-1).float()
        treatment_cur = treatment[torch.arange(len(treatment)), (torch.sum(src_mask, dim=-1) - 1)]

        for t in range(trg_len):
            # insert input token embedding, previous hidden state and the context state
            # receive output tensor (predictions) and new hidden state
            treatment_next = treatment[:, src_len+t]

            (output, output_cf), (hidden, cell), patient_rep, ps_output, decoder_attention = self.decoder(input,
                                                                                                          x_static,
                                                                                                          treatment_cur,
                                                                                                          treatment_next,
                                                                                                          hidden, cell,
                                                                                                          encoder_outputs,
                                                                                                          src_mask)

            treatment_cur = treatment[:, src_len + t]
            # place predictions in a tensor holding predictions for each token
            outputs[:, t] = output
            outputs_cf[:, t] = output_cf
            # decoder_attentions[:, t] = decoder_attention
            decoder_attentions[:, t] = decoder_attention
            ps_outputs[:, t+1] = ps_output

            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            # get the highest predicted token from our predictions
            # top1 = output.argmax(1).float()
            top1 = output.squeeze(-1)

            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[:, t+1] if teacher_force else top1


        return outputs.squeeze(-1), outputs_cf.squeeze(-1), patient_representations, ps_outputs, decoder_attentions
