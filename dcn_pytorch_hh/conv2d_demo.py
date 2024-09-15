import torch


def easy_conv1(img, kernal, step = 1):
    N = img.shape[0]            # 图片边长
    F = kernal.shape[0]         # filter长
    L = int((N-F)/step) + 1     # 输出结果边长
    res = torch.zeros(L, L, requires_grad=False)
    for row in range(0, L):
        for column in range(0, L):
            tmp_row, tmp_col = row * step, column * step
            res[row, column] = (img[tmp_row : tmp_row + F, tmp_col : tmp_col + F] * kernal).sum().item()
    return res


if __name__ == '__main__':
    X = torch.arange(1,17).view(4,4)
    ker = torch.tensor([[1,1],[1,1]])

    print("+-"*25)
    print(easy_conv1(X, ker, 1))
    print("=="*25)
    print(easy_conv1(X, ker, 2))