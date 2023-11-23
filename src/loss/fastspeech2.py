import torch
from torch import nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, mel_output, duration_predictor_output, pitch_predictor_output, 
                energy_predictor_output, mel_target, length_target,
                energy_target, pitch_target, **kwargs):
        
        mel_loss = self.mse_loss(mel_output, mel_target)

        duration_predictor_loss = self.mse_loss(duration_predictor_output, torch.log1p((length_target).float())) 

        energy_predictor_loss = self.mse_loss(energy_predictor_output, torch.log1p(energy_target))

        pitch_predictor_loss = self.mse_loss(pitch_predictor_output, torch.log1p(pitch_target))

        return mel_loss, duration_predictor_loss, energy_predictor_loss, pitch_predictor_loss