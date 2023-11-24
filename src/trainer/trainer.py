import random
from pathlib import Path
from random import shuffle

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker
from src.text import text_to_sequence
from src.waveglow.inference import get_wav


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """  
        for tensor_for_gpu in ["mel_target", "mel_pos", "src_seq", "src_pos", "pitch_target", "energy_target", "duration_target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        print('qq')
        print(self.train_dataloader)
                
        for data in self.train_dataloader:
            print(data)
            break
        print(len(self.train_dataloader))


        progress_bar = tqdm(range(self.len_epoch), desc='train')
        print('qq')

        for list_batch_idx, list_batch in enumerate(self.train_dataloader):

            print('qq')
            for data in self.train_dataloader:
                print(data)
                break

            stop = False
            for batch_idx, batch in enumerate(list_batch):
                progress_bar.update(1)
                
                print('qq')
                for data in self.train_dataloader:
                    print(data)
                    break

                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics,
                        index=batch_idx,
                        total=self.len_epoch,
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                    
                print('qq2')

                full_batch_idx = batch_idx + list_batch_idx * self.batch_expand_size
                if full_batch_idx % self.log_step == 0:
                    self.writer.set_step((epoch - 1) * self.len_epoch + full_batch_idx)
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(full_batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )

                    self._log_scalars(self.train_metrics)

                    last_train_metrics = self.train_metrics.result()
                    self.train_metrics.reset()
                if full_batch_idx >= self.len_epoch:
                    stop = True
                    break
            if stop:
                break
        log = last_train_metrics

        return log


    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, index):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)


        mel_loss, duration_loss, energy_loss, pitch_loss = self.criterion(**batch)
        batch['mel_loss'] = mel_loss
        batch['duration_loss'] = duration_loss
        batch['energy_loss'] = energy_loss
        batch['pitch_loss'] = pitch_loss
        batch['loss'] = mel_loss + duration_loss + pitch_loss + energy_loss
        if is_train:
            batch['loss'].backward()
            if (index + 1) % self.batch_accum_steps == 0 or index + 1 == self.len_epoch:
                self._clip_grad_norm()
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()


        metrics.update('mel_loss', batch['mel_loss'].item())
        metrics.update('duration_loss', batch['duration_loss'].item())
        metrics.update('energy_loss', batch['energy_loss'].item())
        metrics.update('pitch_loss', batch['pitch_loss'].item())
        metrics.update('loss', batch['loss'].item())

        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    
    def _log_audio(self, name, audio, sr):
        self.writer.add_audio(f"audio{name}", audio, sample_rate=sr)
        

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
