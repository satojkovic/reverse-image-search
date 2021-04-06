import torch
from torch import nn
import torch.nn.functional as F


class TrainerBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        loss = 1 - F.cosine_similarity(out, targets).mean()
        return loss

    def validation_step(self, batch):
        images, targets = batch
        out = self(images)
        loss = 1 - F.cosine_similarity(out, targets).mean()
        return {'val_loss': loss.detach()}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        return {'val_loss': epoch_loss.item()}

    def epoch_end(self, epoch, result):
        print('Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}'.format(
            epoch, result['train_loss'], result['val_loss']
        ))


if __name__ == "__main__":
    pass
