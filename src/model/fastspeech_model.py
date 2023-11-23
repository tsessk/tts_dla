import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):
        '''
        Calculates scaled dot product attention for given quires, keys and values.

        Parameters
        ----------
        q, k, v : Tensor((batch_size * n_heads) x seq_len x hidden_size)
        mask : Tensor((batch_size * n_heads) x seq_len x seq_len)
        Returns
        -------
        output : Tensor((batch_size * n_heads) x seq_len x hidden_size)
        attn : Tensor((batch_size * n_heads) x seq_len x seq_len)
        '''

        attn = torch.bmm(q, k.transpose(-1, -2))
        attn = attn / self.temperature
        
        if mask is not None:
            attn = attn.masked_fill(mask, -torch.inf)
        
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = attn @ v
        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5) 
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.w_qs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0,
                        std=np.sqrt(2.0 / (self.d_model + self.d_v))) 

    def forward(self, q, k, v, mask=None):
            d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

            sz_b, len_q, _ = q.size()
            sz_b, len_k, _ = k.size()
            sz_b, len_v, _ = v.size()

            residual = q

            q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
            k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
            v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

            q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
            k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
            v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv
            
            if mask is not None:
                mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..
            output, attn = self.attention(q, k, v, mask=mask)

            output = output.view(n_head, sz_b, len_q, d_v)
            output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

            output = self.dropout(self.fc(output))
            output = self.layer_norm(output + residual)

            return output, attn
        

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=fft_conv1d_kernel[0], padding=fft_conv1d_padding[0])

        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=fft_conv1d_kernel[1], padding=fft_conv1d_padding[1])

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = self.layer_norm(x).transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = output + residual

        return output


class FFTBlock(nn.Module):
    '''FFT Block'''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, fft_conv1d_kernel, fft_conv1d_padding, dropout=0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel, fft_conv1d_padding, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        
        if non_pad_mask is not None:
            enc_output *= non_pad_mask

        return enc_output, enc_slf_attn
    

def create_alignment(base_mat, predictor_output):
    N, L = predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + predictor_output[i][j]
    return base_mat


class Transpose(nn.Module):
    def __init__(self, dim_1, dim_2):
        super().__init__()
        self.dim_1 = dim_1
        self.dim_2 = dim_2

    def forward(self, x):
        return x.transpose(self.dim_1, self.dim_2)


class VarPredictor(nn.Module):
    """ Duration/Pitch/Energy Predictor """

    def __init__(self, encoder_dim, predictor_filter_size, predictor_kernel_size, dropout):
        super().__init__()

        self.input_size = encoder_dim
        self.filter_size = predictor_filter_size
        self.kernel = predictor_kernel_size
        self.conv_output_size = predictor_filter_size
        self.dropout = dropout

        self.conv_net = nn.Sequential(
            Transpose(-1, -2),
            nn.Conv1d(
                self.input_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            Transpose(-1, -2),
            nn.Conv1d(
                self.filter_size, self.filter_size,
                kernel_size=self.kernel, padding=1
            ),
            Transpose(-1, -2),
            nn.LayerNorm(self.filter_size),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)
        self.relu = nn.ReLU()

    def forward(self, encoder_output):
        encoder_output = self.conv_net(encoder_output)
            
        out = self.linear_layer(encoder_output)
        out = self.relu(out)
        out = out.squeeze()
        if not self.training:
            out = out.unsqueeze(0)
        return out
    
class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self, encoder_dim, predictor_filter_size, predictor_kernel_size, dropout):
        super(LengthRegulator, self).__init__()
        self.duration_predictor = VarPredictor(encoder_dim, predictor_filter_size, predictor_kernel_size, dropout)

    def LR(self, x, duration_predictor_output, mel_max_length=None):
        expand_max_len = torch.max(
            torch.sum(duration_predictor_output, -1), -1)[0]
        alignment = torch.zeros(duration_predictor_output.size(0),
                                expand_max_len,
                                duration_predictor_output.size(1)).numpy()
        alignment = create_alignment(alignment,
                                     duration_predictor_output.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_length:
            output = F.pad(
                output, (0, 0, 0, mel_max_length-output.size(1), 0, 0))
        return output

    def forward(self, x, alpha=1.0, target=None, mel_max_length=None):
        duration_predictor_output = self.duration_predictor(x)
        if target is not None:
            output = self.LR(x, target, mel_max_length)
            return output, duration_predictor_output
        else:
            duration_predictor_output = (((torch.exp(duration_predictor_output) - 1) * alpha) + 0.5).int()
            output = self.LR(x, duration_predictor_output, mel_max_length)

            mel_pos = torch.stack([torch.Tensor([i+1  for i in range(output.size(1))])]
                ).long().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            return output, mel_pos
        

def get_non_pad_mask(seq, PAD):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_attn_key_pad_mask(seq_k, seq_q, PAD):
    ''' For masking out the padding part of key sequence. '''
    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(
        1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask


class Encoder(nn.Module):
    def __init__(self, max_seq_len, encoder_n_layer, vocab_size, encoder_dim, PAD, encoder_conv1d_filter_size,
                 fft_conv1d_kernel, fft_conv1d_padding, encoder_head, dropout):
        super(Encoder, self).__init__()
        
        len_max_seq= max_seq_len
        n_position = len_max_seq + 1
        n_layers = encoder_n_layer

        self.pad = PAD

        self.src_word_emb = nn.Embedding(
            vocab_size,
            encoder_dim,
            padding_idx=PAD
        )

        self.position_enc = nn.Embedding(
            n_position,
            encoder_dim,
            padding_idx=PAD
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            encoder_dim,
            encoder_conv1d_filter_size,
            encoder_head,
            encoder_dim // encoder_head,
            encoder_dim // encoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, src_seq, src_pos, return_attns=False):

        enc_slf_attn_list = []

        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq, PAD=self.pad)
        non_pad_mask = get_non_pad_mask(src_seq, PAD=self.pad)
        
        enc_output = self.src_word_emb(src_seq) + self.position_enc(src_pos)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]
        

        return enc_output, non_pad_mask
    

class Decoder(nn.Module):
    """ Decoder """

    def __init__(self, max_seq_len, decoder_n_layer, decoder_dim, PAD, decoder_conv1d_filter_size,
                             fft_conv1d_kernel, fft_conv1d_padding, decoder_head, dropout):

        super(Decoder, self).__init__()

        len_max_seq = max_seq_len
        n_position = len_max_seq + 1
        n_layers = decoder_n_layer

        self.pad = PAD

        self.position_enc = nn.Embedding(
            n_position,
            decoder_dim,
            padding_idx=PAD,
        )

        self.layer_stack = nn.ModuleList([FFTBlock(
            decoder_dim,
            decoder_conv1d_filter_size,
            decoder_head,
            decoder_dim // decoder_head,
            decoder_dim // decoder_head,
            fft_conv1d_kernel,
            fft_conv1d_padding,
            dropout=dropout
        ) for _ in range(n_layers)])

    def forward(self, enc_seq, enc_pos, return_attns=False):
        dec_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=enc_pos, seq_q=enc_pos, PAD=self.pad)
        non_pad_mask = get_non_pad_mask(enc_pos, PAD=self.pad)

        # -- Forward
        dec_output = enc_seq + self.position_enc(enc_pos)

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]

        return dec_output
    

def get_mask_from_lengths(lengths, max_len=None):
    if max_len == None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len, 1, device=lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool()

    return mask


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""
    def __init__(
        self, encoder_dim, pitch_predictor_filter_size, pitch_predictor_kernel_size,
        energy_predictor_filter_size, energy_predictor_kernel_size,
        dur_predictor_filter_size, dur_predictor_kernel_size, num_embed,
         min_pitch, max_pitch, min_energy, max_energy, dropout):
        super().__init__()

        self.pitch_predictor = VarPredictor(encoder_dim, pitch_predictor_filter_size,
                                             pitch_predictor_kernel_size, dropout)
        
        self.energy_predictor = VarPredictor(encoder_dim, energy_predictor_filter_size,
                                             energy_predictor_kernel_size, dropout)
        
        self.length_regulator = LengthRegulator(encoder_dim, dur_predictor_filter_size,
                                                 dur_predictor_kernel_size, dropout)

        self.pitch_embedding = nn.Embedding(num_embed, encoder_dim)
        self.energy_embedding = nn.Embedding(num_embed, encoder_dim)

        self.pitch_buckets = torch.linspace(min_pitch, max_pitch, num_embed - 1)
        self.energy_buckets = torch.linspace(min_energy, max_energy, num_embed - 1)

    def forward(self, encoder_output, len_coef=1.0, pitch_coef=1.0, energy_coef=1.0, length_target=None,
        pitch_target=None, energy_target=None, mel_max_length=None):

        lr_output, duration_pred = self.length_regulator(encoder_output, len_coef, length_target, mel_max_length)
        pitch_pred = self.pitch_predictor(lr_output)
        energy_pred = self.energy_predictor(lr_output)

        if self.training:
            pitch_embeddings = self.pitch_embedding(torch.bucketize(pitch_target, self.pitch_buckets))
            energy_embeddings = self.energy_embedding(torch.bucketize(energy_target, self.energy_buckets))

            result = lr_output + pitch_embeddings + energy_embeddings
            return result, duration_pred, pitch_pred, energy_pred
        else:
            pitch_prediction = (torch.exp(duration_pred) - 1) * pitch_coef
            pitch_embeddings = self.pitch_embedding(torch.bucketize(pitch_prediction, self.pitch_buckets))

            energy_prediction = (torch.exp(energy_pred) - 1) * energy_coef
            energy_embeddings = self.energy_embedding(torch.bucketize(energy_prediction, self.energy_buckets))

            result = lr_output + pitch_embeddings + energy_embeddings

            return result, duration_pred


class FastSpeech2(nn.Module):
    """ FastSpeech """

    def __init__(self, max_seq_len, encoder_n_layer, vocab_size, encoder_dim,
                 PAD, encoder_conv1d_filter_size, encoder_head, decoder_n_layer, decoder_dim,
                 decoder_conv1d_filter_size, decoder_head,
                 fft_conv1d_kernel, fft_conv1d_padding, predictor_filter_size, predictor_kernel_size,
                 pitch_predictor_filter_size, pitch_predictor_kernel_size, energy_predictor_filter_size,
                energy_predictor_kernel_size, dur_predictor_filter_size, dur_predictor_kernel_size,
                num_embed, min_pitch, max_pitch, min_energy, max_energy,
                num_mels, dropout):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder(max_seq_len, encoder_n_layer, vocab_size, encoder_dim,
                 PAD, encoder_conv1d_filter_size,
                 fft_conv1d_kernel, fft_conv1d_padding, encoder_head, dropout) 
        
        self.length_regulator = LengthRegulator(encoder_dim, predictor_filter_size, predictor_kernel_size, dropout)

        self.decoder = Decoder(max_seq_len, decoder_n_layer, decoder_dim, PAD, decoder_conv1d_filter_size,
                             fft_conv1d_kernel, fft_conv1d_padding, decoder_head, dropout)

        self.mel_linear = nn.Linear(decoder_dim, num_mels)

        self.varadaptor = VarianceAdaptor(encoder_dim, pitch_predictor_filter_size, pitch_predictor_kernel_size,
        energy_predictor_filter_size, energy_predictor_kernel_size,
        dur_predictor_filter_size, dur_predictor_kernel_size, num_embed,
        min_pitch, max_pitch, min_energy, max_energy, dropout)

    def mask_tensor(self, mel_output, position, mel_max_length):
        lengths = torch.max(position, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_length)
        mask = mask.unsqueeze(-1).expand(-1, -1, mel_output.size(-1))
        return mel_output.masked_fill(mask, 0.)

    def forward(self, src_seq, src_pos, mel_pos=None, mel_max_length=None, length_target=None, pitch_target=None,
                energy_target=None, len_coeff=1.0, pitch_coeff=1.0, energy_coeff=1.0):
        enc_output, non_pad_mask = self.encoder(src_seq, src_pos)

        if self.training:
            lr_output, duration_predictor_output, pitch_predictor_output, energy_predictor_output = self.variance_adaptor(enc_output,
                len_coeff, pitch_coeff, energy_coeff, length_target, pitch_target, energy_target, mel_max_length)
            
            output = self.decoder(lr_output, mel_pos)
            output = self.mel_linear(output)
            output = self.mask_tensor(output, mel_pos, mel_max_length)

            #pitch_predictor_output = self.mask_tensor(pitch_predictor_output.unsqueeze(-1), mel_pos, mel_max_length).squeeze() 

            #energy_predictor_output = self.mask_tensor(energy_predictor_output.unsqueeze(-1), mel_pos, mel_max_length).squeeze() 

            return output, duration_predictor_output, pitch_predictor_output, energy_predictor_output
        else:
            lr_output, duration_predictor_output = self.variance_adaptor(enc_output, len_coeff, pitch_coeff, energy_coeff)
            output = self.decoder(lr_output, duration_predictor_output)
            output = self.mel_linear(output)
            return output


