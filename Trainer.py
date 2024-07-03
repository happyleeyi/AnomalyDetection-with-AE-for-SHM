import time
import torch.optim as optim
import torch #파이토치 기본모듈
from Autoencoder import Autoencoder

class train_model:
    def __init__(self, lr, weight_decay, epochs, device, train_loader, rep_dim, pretrained):
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.device = device
        self.train_loader = train_loader
        self.rep_dim = rep_dim
        self.pretrained = pretrained

    def train(self):
        ae_net = Autoencoder(self.rep_dim).to(self.device)

        if self.pretrained:
            ae_net.load_state_dict(torch.load('aefloormodel_rep4_state_dict.pt'))
        else:
            lr_milestones = tuple()
            optimizer = optim.Adam(ae_net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                        amsgrad='adam'=='amsgrad')
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=0.1)

            print('Starting pretraining...')
            start_time = time.time()
            ae_net.train()
            for epoch in range(self.epochs):

                scheduler.step()
                if epoch in lr_milestones:
                    print('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

                loss_epoch = 0.0
                n_batches = 0
                epoch_start_time = time.time()
                for X, Y in self.train_loader:
                    X = X.to(self.device)

                    # Zero the network parameter gradients
                    optimizer.zero_grad()

                    # Update network parameters via backpropagation: forward + backward + optimize
                    outputs = ae_net(X)
                    scores = torch.sum((outputs - X) ** 2, dim=tuple(range(1, outputs.dim())))
                    loss = torch.mean(scores)
                    loss.backward()
                    optimizer.step()

                    loss_epoch += loss.item()
                    n_batches += 1

                # log epoch statistics
                epoch_train_time = time.time() - epoch_start_time
                print('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                            .format(epoch + 1, self.epochs, epoch_train_time, loss_epoch / n_batches))

            pretrain_time = time.time() - start_time
            print('Pretraining time: %.3f' % pretrain_time)
            print('Finished pretraining.')

            torch.save(ae_net.state_dict(), 'aefloormodel_state_dict.pt')

        self.model = ae_net
        return self.model