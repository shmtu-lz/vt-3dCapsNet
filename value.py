import timeit
import os
import glob
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 500  # 训练次数
resume_epoch = 3  # 默认值为0，如果想继续，请更改
useTest = True # 测试使用
nTestInterval = 20 # 在测试集上运行每个nTestInterval时间点
snapshot = 50 # 每个快照时代存储一个模型
lr = 1e-4 # 学习率

dataset = 'ucf101' # 使用ucf101数据集
num_classes = 7


save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'C3D' # 模型名称
saveName = modelName + '-' + dataset


def train_model(dataset=dataset, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs,  useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): 分类数：101
            num_epochs (int, optional):训练好的次数
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=False)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    print(num_classes)
    model = torch.load('./pretrain_model/net94.pkl')

    model.to(device)
    criterion.to(device)


    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train'), batch_size=80, shuffle=True, num_workers=0)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val'), batch_size=80, num_workers=0)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test'), batch_size=80, num_workers=0)

    trainval_loaders = {'train': train_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train']}
    best_acc = 0.9
    for epoch in range(0, num_epochs):
        # 进行训练集训练和验证机验证
        for phase in ['train']:
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            # 模型的模式选择


            for _,(inputs, labels) in enumerate(trainval_loaders[phase]):
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                outputs,out_cap = model(inputs)


                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]

                labels = torch.tensor(labels, dtype=torch.long)


                running_corrects += torch.sum(preds == labels)


            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]



            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")




if __name__ == "__main__":
    train_model()