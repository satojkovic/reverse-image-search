import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from tqdm import tqdm


class TrainerBase(nn.Module):
    def training_step(self, batch):
        images, targets = batch
        out = self(images)
        # Not same as the orignal paper
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


class DeviseModel(TrainerBase):
    def __init__(self, embed_dim=300):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        # Replace last layer
        num_features = self.network.fc.in_features
        self.network.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=num_features,
                      out_features=num_features // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features // 2),
            nn.Dropout(p=0.5),
            nn.Linear(num_features//2, embed_dim)
        )

    def forward(self, xb):
        return self.network(xb)


@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            # Traininig phase
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history


if __name__ == "__main__":
    pass
