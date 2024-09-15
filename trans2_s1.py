import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch.optim as optim
import time
from collections import OrderedDict
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from config import *
from model_1 import *
# from utils.load_GDD import *
from gjh.utils.Miou import *
from gjh.utils.loss import *
from gjh.utils.dataloader import *

train_min_loss = float('inf')
valid_max_glass = float(0)

scale = 384

parser = argparse.ArgumentParser(description="PyTorch Glass Detection Example")
parser.add_argument("--batchSize", type=int, default=12, help="Training batch size")
parser.add_argument("--epochs", type=int, default=200, help='')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--use_GPU", type=bool, default=True, help='')
parser.add_argument("--data_path", type=str, default="/data/GDD", help='')
# parser.add_argument("--data_path", type=str, default="/data/gjh2/GEGD/", help='')
parser.add_argument("--save_path", type=str, default="./logs/Ver.0.4.12.s3", help='')  # 88.95
parser.add_argument("--lr", type=float, default=1e-5, help='')

opt = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

if not os.path.isdir(opt.save_path):
    os.makedirs(opt.save_path)

rgb_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor()
])

grey_transform = transforms.Compose([
    transforms.Resize((scale, scale)),
    transforms.ToTensor()
])

# load dataset
print('INFO:Loading dataset ...\n')
dataset_train = make_dataSet(opt.data_path, train=True, rgb_transform=rgb_transform, grey_transform=grey_transform)
dataset_valid = make_dataSet(opt.data_path, train=False, rgb_transform=rgb_transform, grey_transform=grey_transform)

loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True, drop_last=True)
loader_valid = DataLoader(dataset=dataset_valid, num_workers=4, batch_size=opt.batchSize, shuffle=False)
print("# of training samples: %d\n" % int(len(dataset_train)))
print("# of valid samples: %d\n" % int(len(dataset_valid)))


model = Net_s3(backbone_path=swin_B)
model = model.cuda()
'''
# ##########create model###############
param_optim=[]
count = 0 
for name, module in model._modules.items():
    count = count +1
    if count <= 23:
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            param_optim.append(p)
            
param_groups = [{'params': param_optim}]
optimizer = torch.optim.Adam(param_groups, lr=opt.lr)
    # print(name)
    # print(count)
    # count = count +1
'''

# ##### optim #######

param_optim = []
for name, module in model._modules.items():
    if name == "SEM3" or name == "SEM2" or name == "SEM1" or name == "SEM0":
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            param_optim.append(p)

param_groups = [{'params': param_optim}]
optimizer = torch.optim.Adam(param_groups, lr=opt.lr)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)

# ##### Loss #######
criterion = nn.L1Loss()
MseCriterion = nn.MSELoss()
bce = nn.BCELoss()

for epoch in range(opt.epochs):
    # ####### train ########
    start = time.time()
    model.train()
    model.zero_grad()

    glass_loss_sum = 0
    train_loss_sum = 0
    train_r1_sum = 0
    train_r2_sum = 0
    train_ghost_sum = 0
    train_shift_sum = 0
    train_glass_sum = 0
    t_glass_iou = 0

    for idx, (input_data, glass) in enumerate(loader_train, 0):
        input_data = Variable(input_data)
        glass = Variable(glass)

        if opt.use_GPU:
            input_data = input_data.cuda()
            glass = glass.cuda()

        optimizer.zero_grad()

        g3, g2, g1, g0, fg, l3, l2, l1, l0, final_glass = model(input_data)
        
        lossg3 = bce_ssim_loss(g3, glass)
        lossg2 = bce_ssim_loss(g2, glass)
        lossg1 = bce_ssim_loss(g1, glass)
        lossg0 = bce_ssim_loss(g0, glass)
        lossgfinal = bce_ssim_loss(fg, glass)
        
        lossGhost = lossg3 + lossg2 + lossg1 + lossg0 + lossgfinal

        loss3 = bce_ssim_loss(l3, glass)
        loss2 = bce_ssim_loss(l2, glass)
        loss1 = bce_ssim_loss(l1, glass)
        loss0 = bce_ssim_loss(l0, glass)
        lossfinal = bce_ssim_loss(final_glass, glass)

        lossGlass = lossfinal + loss3 + loss2 + loss1 + loss0

        loss = lossGlass
        loss.backward()

        train_glass_sum += lossGlass.item()
        train_ghost_sum += lossGhost.item()
        optimizer.step()

    # ###### Validing ######

    model.eval()
    model.zero_grad()
    v_ghost_iou = 0
    v_glass_iou = 0
    valid_loss_sum = 0

    with torch.no_grad():
        for idx, (input_data, glass) in enumerate(loader_valid, 0):
            input_data = Variable(input_data)
            glass = Variable(glass)
            if opt.use_GPU:
                input_data = input_data.cuda()
                glass = glass.cuda()

            optimizer.zero_grad()

            g3, g2, g1, g0, fg, l3, l2, l1, l0, final_glass = model(input_data)

            pred = final_glass
            label = glass
            bs, _, _, _ = label.shape

            temp1 = pred.data.squeeze(1)
            temp2 = label.data.squeeze(1)
            for i in range(bs):
                a = temp1[i, :, :]
                b = temp2[i, :, :]
                a = torch.round(a).squeeze(0).int().detach().cpu()
                b = torch.round(b).squeeze(0).int().detach().cpu()
                v_glass_iou += iou_mean(a, b, 1)

            torch.cuda.empty_cache()

    end = time.time()
    t = end - start

    print('INFO: epoch:{},tgl:{},tgsl:{},V_gh:{},V_gl:{}, time:{}'.format(epoch + 1,
                                                                  round(train_glass_sum / len(loader_train), 6),
                                                                  round(train_ghost_sum / len(loader_train), 6),
                                                                  round(
                                                                      v_ghost_iou / len(
                                                                          loader_valid) / opt.batchSize * 100,
                                                                      2),
                                                                  round(
                                                                      v_glass_iou / len(
                                                                          loader_valid) / opt.batchSize * 100,
                                                                      2),
                                                                  round(t, 2)))

    # del temporary outputs and loss
    # del final_glass, lossGlass, loss1, loss2, loss3, loss0, lossfuse, lossfinal

    if train_loss_sum < train_min_loss:
        train_min_loss = train_loss_sum
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'train_min.pth'))
        print("INFO: save train_min model")

    if v_glass_iou > valid_max_glass:
        valid_max_glass = v_glass_iou
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'glass_max.pth'))
        print("INFO: save glass model")

    # if epoch % 10 == 0 :
    # name = 'epoch' + str(epoch) + '.pth'
    # torch.save(model.state_dict(), os.path.join(opt.save_path, name))
    # print("INFO: save model")
