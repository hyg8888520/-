from utils_for_dataset import getDataLoader
from utils_train_valid import Trainer

from ZhuNet import Zhu_Net

import time
import torch
import time
import torch

train_loader, valid_loader, test_loader = getDataLoader(
    r'D:\cv\big shuju\images\train_images',# cover训练集
    r'D:\cv\big shuju\images\train_stego',# stego训练集
    r'D:\cv\big shuju\fujian1\gray\orige',# cover验证集
    r'D:\cv\big shuju\fujian1\gray\me',# stego验证集
    r'D:\cv\big shuju\fujian2\gray',  # cover测试集
    r'D:\cv\big shuju\fujian2\gray',  # stego测试集
    1
)
net = Zhu_Net() # 无预训练模型
# model = torch.load('model100p.pth')
# model.load_state_dict('70acc_38.pkl')


trainer = Trainer(model=net, lr=0.001, cur_epoch=0, lr_decay=0.95, weight_decay=0.0, shedule_lr=[20, 35, 50, 65],
                  token='Best_biggest_hill_0.1', token1='hill_0.1_trueD', save_dir="Best_biggest_hill_0.1_trueD",
                  print_freq=150)

for cur_epoch in range(100):
    time1 = time.time()
    trainer.train(train_loader=train_loader)
    time2 = time.time()
    print("epoch time: {}".format(time2 - time1))
    trainer.valid(valid_loader=valid_loader)
    trainer.save_loss_val_acc()
    time3 = time.time()
    print("valid time: {}".format(time3 - time2))
torch.save(net, r'PATH' + '.pth')

# trainer.test(test_loader)