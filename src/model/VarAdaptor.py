from torch import nn
import torch
import torch.nn.functional as F


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