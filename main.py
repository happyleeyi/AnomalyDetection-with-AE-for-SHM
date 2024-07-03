import torch #파이토치 기본모듈
from Dataset import get_data
from Variables import path_damaged, path_undamaged, BATCH_SIZE, rep_dim, lr, epochs, weight_decay, num_class, data_saved, pretrained, threshold_quantile
from Trainer import train_model
from Tester import test_model


if torch.cuda.is_available():
    device = torch.device('cuda') #GPU이용

else:
    device = torch.device('cpu') #GPU이용안되면 CPU이용

print('Using PyTorch version:', torch.__version__, ' Device:', device)

Data = get_data(path_undamaged, path_damaged)
train_loader, test_loader = Data.load_data(BATCH_SIZE, data_saved)

SVDD_trainer = train_model(lr, weight_decay, epochs, device, train_loader, rep_dim, pretrained)
model = SVDD_trainer.train()

for th in threshold_quantile:
    SVDD_tester = test_model(model, test_loader, train_loader, num_class, th, device)
    SVDD_tester.confusion_mat()



