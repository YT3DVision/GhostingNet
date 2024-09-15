import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import argparse
import torch.optim as optim
import time
import os

from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader

from config import *
from model_shift3 import *
# from utils.load_GDD import *
from utils.Miou import *
from utils.loss import *
from utils.data_w_shift import *

train_min_loss = float('inf')
valid_max_glass = float(0)
valid_max_ghost = float(0)

scale = 384

parser = argparse.ArgumentParser(description="PyTorch Glass Detection Example")
parser.add_argument("--batchSize", type=int, default=4, help="Training batch size")
parser.add_argument("--epochs", type=int, default=200, help='')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--use_GPU", type=bool, default=True, help='')
# parser.add_argument("--data_path", type=str, default="/data/gjh2/GEGD", help='')
parser.add_argument("--data_path", type=str, default="/data/dataSet/synthetic/", help='') # synthetic
parser.add_argument("--save_path", type=str, default="./logs/Ver0.4.10.s1", help='')  # 88.95
parser.add_argument("--lr", type=float, default=1e-5, help='')

opt = parser.parse_args()

pz3 = torch.zeros(opt.batchSize, 1024, 12, 12).cuda()
pz2 = torch.zeros(opt.batchSize, 512, 24, 24).cuda()
pz1 = torch.zeros(opt.batchSize, 256, 48, 48).cuda()
pz0 = torch.zeros(opt.batchSize, 128, 96, 96).cuda()

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

# ##########create model###############
model = Net(backbone_path=swin_B)
if opt.use_GPU:
    model = model.cuda()

# ##### optim #######
'''
param_optim = []
for name, module in model._modules.items():
    print(name)
    if name != "SEM3" and name != "SEM2" and name != "SEM1" and name != "SEM0":
        for p in module.parameters():
            p.requires_grad = False
    else:
        for p in module.parameters():
            param_optim.append(p)

param_groups = [{'params': param_optim}]
optimizer = torch.optim.Adam(param_groups, lr=opt.lr)
'''
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
    train_w_sum = 0
    train_h_sum = 0
    train_ghost_sum = 0
    train_shift_sum = 0
    train_glass_sum = 0
    train_per_sum = 0
    t_glass_iou = 0
    t_ghost_iou = 0

    for idx, (input_data, glass, ghost, r1, r2, h, w) in enumerate(loader_train, 0):
        input_data = Variable(input_data)
        ghost = Variable(ghost)
        glass = Variable(glass)
        r1 = Variable(r1)
        r2 = Variable(r2)
        h = Variable(h)
        w = Variable(w)

        if opt.use_GPU:
            input_data = input_data.cuda()
            ghost = ghost.cuda()
            glass = glass.cuda()
            r1 = r1.cuda()
            r2 = r2.cuda()
            h = h.cuda()
            w = w.cuda()

        optimizer.zero_grad()
        
        h3, h2, h1, h0, \
        w3, w2, w1, w0, \
        per3, per2, per1, per0, \
        r1_3, r1_2, r1_1, r1_0, \
        r2_3, r2_2, r2_1, r2_0, \
        g3, g2, g1, g0, fg = model(input_data)
        
        lossh3 = MseCriterion(h3, h)
        lossh2 = MseCriterion(h2, h)
        lossh1 = MseCriterion(h1, h)
        lossh0 = MseCriterion(h0, h)

        lossH = lossh3 + lossh2 + lossh1 + lossh0

        lossw3 = MseCriterion(w3, w)
        lossw2 = MseCriterion(w2, w)
        lossw1 = MseCriterion(w1, w)
        lossw0 = MseCriterion(w0, w)

        lossW = lossw3 + lossw2 + lossw1 + lossw0
        
        lossp3 = MseCriterion(per3, pz3)
        lossp2 = MseCriterion(per2, pz2)
        lossp1 = MseCriterion(per1, pz1)
        lossp0 = MseCriterion(per0, pz0)

        lossPer = lossp3 + lossp2 + lossp1 + lossp0

        lossr13 = bce(r1_3, r1)
        lossr12 = bce(r1_2, r1)
        lossr11 = bce(r1_1, r1)
        lossr10 = bce(r1_0, r1)

        lossR1 = lossr13 + lossr12 + lossr11 + lossr10

        lossr23 = bce(r2_3, r2)
        lossr22 = bce(r2_2, r2)
        lossr21 = bce(r2_1, r2)
        lossr20 = bce(r2_0, r2)

        lossR2 = lossr23 + lossr22 + lossr21 + lossr20

        lossg3 = bce(g3, ghost)
        lossg2 = bce(g2, ghost)
        lossg1 = bce(g1, ghost)
        lossg0 = bce(g0, ghost)
        lossgfuse = bce(fg, ghost)

        lossGhost = lossgfuse + lossg3 + lossg2 + lossg1 + lossg0
        
        loss = lossR1 + lossR2 + 2 * lossGhost + 0.1 * lossPer + 10 * lossH + 10 * lossW 
        loss.backward()
        
        train_loss_sum += loss.item()
        train_per_sum += lossPer.item()
        train_r1_sum += lossR1.item()
        train_r2_sum += lossR2.item()
        train_w_sum += lossW.item()
        train_h_sum += lossH.item()
        train_ghost_sum += lossGhost.item()
        # train_glass_sum += lossGlass.item()
        
        optimizer.step()

        torch.cuda.empty_cache()

    # ###### Validing ######

    model.eval()
    model.zero_grad()
    v_ghost_iou = 0
    v_glass_iou = 0
    valid_loss_sum = 0

    with torch.no_grad():
        for idx, (input_data, glass, ghost, r1, r2, h, w) in enumerate(loader_valid, 0):
            input_data = Variable(input_data)
            ghost = Variable(ghost)
            glass = Variable(glass)
            r1 = Variable(r1)
            r2 = Variable(r2)
            h = Variable(h)
            w = Variable(w)
    
            if opt.use_GPU:
                input_data = input_data.cuda()
                ghost = ghost.cuda()
                glass = glass.cuda()
                r1 = r1.cuda()
                r2 = r2.cuda()
                h = h.cuda()
                w = w.cuda()

            optimizer.zero_grad()
        
            h3, h2, h1, h0, \
            w3, w2, w1, w0, \
            per3, per2, per1, per0, \
            r1_3, r1_2, r1_1, r1_0, \
            r2_3, r2_2, r2_1, r2_0, \
            g3, g2, g1, g0, fg = model(input_data)
            
            pred = fg
            label = ghost
            bs, _, _, _ = label.shape

            temp1 = pred.data.squeeze(1)
            temp2 = label.data.squeeze(1)
            for i in range(bs):
                a = temp1[i, :, :]
                b = temp2[i, :, :]
                a = torch.round(a).squeeze(0).int().detach().cpu()
                b = torch.round(b).squeeze(0).int().detach().cpu()
                v_ghost_iou += iou_mean(a, b, 1)
                
            '''
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
            '''
            

            torch.cuda.empty_cache()

    end = time.time()
    t = end - start

    print('INFO: epoch:{},tl:{},tpl:{},twl:{},thl:{},tr1:{},tr2:{},tgs:{},tgl:{},V_gh:{},V_gl:{}, time:{}'.format(
        epoch + 1,
        round(train_loss_sum / len(
            loader_train), 6),
        round(train_per_sum / len(
            loader_train), 6),
        round(train_w_sum / len(
            loader_train), 6),
        round(train_h_sum / len(
            loader_train), 6),
        round(train_r1_sum / len(
            loader_train), 6),
        round(train_r2_sum / len(
            loader_train), 6),
        round(train_ghost_sum / len(
            loader_train), 6),
        round(train_glass_sum / len(
            loader_train), 6),
        round(v_ghost_iou / len(
            loader_valid) / opt.batchSize * 100, 2),
        round(v_glass_iou / len(
            loader_valid) / opt.batchSize * 100, 2),
        round(t, 2)))

    # del temporary outputs and loss
    # del final_glass, lossGlass, loss1, loss2, loss3, loss0, lossfuse, lossfinal

    if train_loss_sum < train_min_loss:
        train_min_loss = train_loss_sum
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'train_min.pth'))
        print("INFO: save train_min model")
        
    if v_ghost_iou > valid_max_ghost:
        valid_max_ghost = v_ghost_iou
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'ghost_max.pth'))
        print("INFO: save ghost model")

    # if v_glass_iou > valid_max_glass:
    #     valid_max_glass = v_glass_iou
    #     torch.save(model.state_dict(), os.path.join(opt.save_path, 'glass_max.pth'))
    #     print("INFO: save glass model")


