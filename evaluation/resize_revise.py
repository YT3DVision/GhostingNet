import os
import shutil
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
from skimage import transform

# main_dir = 'D:/DataSet/d40'
# dst_dir = 'D:/GhostDataSet/synthetic/total'

# main_dir = 'G:/contrast2/SCWSSOD'
# dst_dir = 'G:/GEGD_glass_eval/GEGD/SCWSSOD'

main_dir = 'F:/TPAMI/revision_results/GED_Ours_ghost_best/ghost'
dst_dir = 'F:/TPAMI/revision_results/GED_Ours_ghost_best/ghost_resize'

glass_path = os.path.join('D:/GhostDataSet/GEGD_eval_ghost')
# glass_path = os.path.join('D:/GhostDataSet/GEGD_eval')


dst_path = os.path.join(dst_dir)

if not os.path.isdir(dst_path):
    os.makedirs(dst_path)

img_list = [img_name for img_name in os.listdir(glass_path)]
print(img_list)

for idx, img_name in enumerate(img_list):
    print(img_name)
    img = Image.open(os.path.join(main_dir, img_name)).convert('L')
    w, h = img.size
    glass = Image.open(os.path.join(glass_path, img_name[:-4] + ".png"))
    w2, h2 = glass.size

    temp = img.resize((w2, h2), Image.ANTIALIAS)
    print(temp.size)
    temp.save(os.path.join(dst_path, img_name[:-4] + ".png"))

    # shutil.copyfile(os.path.join(main_dir, 'input', img_name), dstfileh)
    # shutil.copyfile(os.path.join(main_dir, 'glass', img_name), dstfileh)
    # shutil.copyfile(os.path.join(main_dir, 'ghost', img_name), dstfileh)
    # shutil.copyfile(os.path.join(main_dir, 'r1', img_name), dstfileh)
    # shutil.copyfile(os.path.join(main_dir, 'r2', img_name), dstfileh)

# image_path = os.path.join(dst_dir)
#
# scale = 384
#
# # parser
# parser = argparse.ArgumentParser(description="PyTorch Mirror Detection Example")
# parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
# parser.add_argument("--use_GPU", type=bool, default=True, help='')
# parser.add_argument("--data_path", type=str, default="/home/dingyuyang/gjh/Shift_dataset/test", help='')
# parser.add_argument("--model_path", type=str, default="./logs/Shift", help='')
# parser.add_argument("--result_path", type=str, default="./result/gdm_w_sem", help='')
# parser.add_argument("--lr", type=float, default=1e-5, help='')
#
# opt = parser.parse_args()
#
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
#
# h_path = os.path.join(opt.result_path, 'h_mat')
# w_path = os.path.join(opt.result_path, 'w_mat')
#
# if not os.path.isdir(opt.result_path):
#     os.makedirs(opt.result_path)
# if not os.path.isdir(h_path):
#     os.makedirs(h_path)
# if not os.path.isdir(w_path):
#     os.makedirs(w_path)
#
# img_transform = transforms.Compose([
#     transforms.Resize((scale, scale)),
#     transforms.ToTensor()
# ])
#
# # ##########create model###############
# model = Net()
# if opt.use_GPU:
#     model = model.cuda()
#
# # ##### Loss #######
# criterion = nn.L1Loss()
# if opt.use_GPU:
#     criterion = criterion.cuda()
#
#
# def test():
#     model.load_state_dict(torch.load(os.path.join(opt.model_path, 'train_min.pth')))
#     model.eval()
#     with torch.no_grad():
#         start = time.time()
#         img_list = [img_name for img_name in os.listdir(os.path.join(opt.data_path, 'input'))]
#         print(img_list)
#
#         for idx, img_name in enumerate(img_list):
#             img = Image.open(os.path.join(opt.data_path, 'input', img_name))
#
#             w, h = img.size
#
#             img_var = Variable(img_transform(img).unsqueeze(0)).cuda()
#
#             out = model(img_var)
#             out_h = out[0].squeeze(0).cpu()
#             out_w = out[1].squeeze(0).cpu()
#
#             np.save(os.path.join(h_path, img_name[:-4] + ".npy"), out_h)
#             np.save(os.path.join(w_path, img_name[:-4] + ".npy"), out_w)
#
#             # np.save(os.path.join(h_path, img_name[:-4] + ".npy"), out_h)
#             # np.save(os.path.join(w_path, img_name[:-4] + ".npy"), out_w)
#
#             # out = out.data.squeeze(0)
#             # out = np.array(transforms.Resize((h, w))(to_pil(out)))
#             #
#             # Image.fromarray(out).save(os.path.join(opt.result_path, img_name[:-4] + ".png"))
#             # Image
