# code originally obtained at:
# https://github.com/elichen/Feature-visualization

import numpy as np
import torch
from torch import tensor
import matplotlib.pyplot as plt
# from IPython.display import clear_output
from torchvision import transforms
import fastai.vision as vision

def init_fft_buf(h, w, device, rand_sd=0.01):
    img_buf = np.random.normal(size=(1, 3, h, w//2 + 1, 2), scale=rand_sd).astype(np.float32)
    spectrum_t = tensor(img_buf).float().cuda(device)
    return spectrum_t

def get_fft_scale(h, w, device, decay_power=.75):
    d=.5**.5 # set center frequency scale to 1
    fy = np.fft.fftfreq(h,d=d)[:,None]
    if w % 2 == 1:
        fx = np.fft.fftfreq(w,d=d)[: w // 2 + 2]
    else:
        fx = np.fft.fftfreq(w,d=d)[: w // 2 + 1]
    freqs = (fx*fx + fy*fy) ** decay_power
    scale = 1.0 / np.maximum(freqs, 1.0 / (max(w, h)*d))
    scale = tensor(scale).float()[None,None,...,None].cuda(device)
    return scale

def fft_to_rgb(h, w, t, device):
    _shape = t.shape
    if len(_shape) == 5:
        scale = get_fft_scale(h, w, device)
        t = scale * t
        t = torch.irfft(t, 2, normalized=True, signal_sizes=(h,w))
    elif len(_shape) == 6:
        for i in range(_shape[2]):
            scale = get_fft_scale(h, w, device)
            t[:, :, i] = scale * t[:, :, i]
        t = torch.irfft(t, 3, normalized=True, signal_sizes=(30,h,w))
    return t

def rgb_to_fft(h, w, t, device):
    _shape = t.shape
    if len(_shape) == 4:
        t = torch.rfft(t, normalized=True, signal_ndim=2)
        scale = get_fft_scale(h, w, device)
        t = t / scale
        
    elif len(_shape) == 5:
        t = torch.rfft(t, normalized=True, signal_ndim=3)  # torch.Size([1, 3, 30, 150, 113, 2])

        for i in range(_shape[2]):
            # t[:, :, i] = torch.rfft(t[:, :, i], normalized=True, signal_ndim=2)
            scale = get_fft_scale(h, w, device)
            t[:, :, i] = t[:, :, i] / scale
    return t

def color_correlation_normalized(device):
    color_correlation_svd_sqrt = np.asarray([[0.26, 0.09, 0.02],
                                             [0.27, 0.00, -0.05],
                                             [0.27, -0.09, 0.03]]).astype(np.float32)
    max_norm_svd_sqrt = np.max(np.linalg.norm(color_correlation_svd_sqrt, axis=0))
    color_correlation_normalized = tensor(color_correlation_svd_sqrt / max_norm_svd_sqrt).cuda(device)
    return color_correlation_normalized

def lucid_colorspace_to_rgb(t, device):
    _shape = t.shape
    if len(_shape) == 4:
        t_flat = t.permute(0,2,3,1)
        t_flat = torch.matmul(t_flat, color_correlation_normalized(device).T)
        t = t_flat.permute(0,3,1,2)
    elif len(_shape) == 5:
        t_flat = t.permute(0,2,3,4,1)
        t_flat = torch.matmul(t_flat, color_correlation_normalized(device).T)
        t = t_flat.permute(0,4,1,2,3)
    return t

def rgb_to_lucid_colorspace(t, device):
    _shape = t.shape

    if len(_shape) == 4:  # n, c, h, w
        t_flat = t.permute(0,2,3,1)  # n, h, w, c
        inverse = torch.inverse(color_correlation_normalized(device).T)
        t_flat = torch.matmul(t_flat, inverse)
        t = t_flat.permute(0,3,1,2)  # n, c, h, w

    elif len(_shape) == 5:  # n, c, d, h, w
        t_flat = t.permute(0,2,3,4,1)  # n, d, h, w, c
        inverse = torch.inverse(color_correlation_normalized(device).T)
        t_flat = torch.matmul(t_flat, inverse)
        t = t_flat.permute(0,4,1,2,3)  # n, c, d, h, w
    return t

def imagenet_mean_std(device):
    return (tensor([0.485, 0.456, 0.406]).cuda(device),
            tensor([0.229, 0.224, 0.225]).cuda(device))

def denormalize(x, device):
    mean, std = imagenet_mean_std(device)
    return x.float()*std[...,None,None] + mean[...,None,None]

def normalize(x, device):
    mean, std = imagenet_mean_std(device)
    _shape = x.shape
    if len(_shape) == 4:
        return (x-mean[...,None,None]) / std[...,None,None]
    elif len(_shape) == 5:
        return (x-mean[...,None,None,None]) / std[...,None,None,None]

def image_buf_to_rgb(h, w, img_buf, device):
    img = img_buf.detach()
    img = fft_to_rgb(h, w, img, device)
    img = lucid_colorspace_to_rgb(img, device)
    img = torch.sigmoid(img)
    img = img[0]
    return img

def show_rgb(img, label=None, ax=None, dpi=25):
    plt_show = True if ax == None else False
    if ax == None: _, ax = plt.subplots(figsize=(img.shape[2]/dpi,img.shape[1]/dpi))
    x = img.cpu().permute(1,2,0).numpy()
    ax.imshow(x)
    ax.axis('off')
    ax.set_title(label)
    if plt_show: plt.show()

def gpu_affine_grid(size, device):
    size = ((1,)+size)

    if len(size) == 4:
        N, C, H, W = size
        grid = torch.FloatTensor(N, H, W, 2).cuda(device)
        linear_points = torch.linspace(-1, 1, W) if W > 1 else tensor([-1.])
        grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])

        linear_points = torch.linspace(-1, 1, H) if H > 1 else tensor([-1.])
        grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
        return vision.FlowField(size[2:], grid)
    elif len(size) == 5:
        N, C, D, H, W = size
        grid = torch.FloatTensor(N, D, H, W, 2).cuda(device)

        linear_points = torch.linspace(-1, 1, W) if W > 1 else tensor([-1.])
        grid[:, :, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, :, 0])

        linear_points = torch.linspace(-1, 1, H) if H > 1 else tensor([-1.])
        grid[:, :, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, :, 1])

        return vision.FlowField(size[2:], grid)

def lucid_transforms(img, device, jitter=None, scale=.5, degrees=45):
    h,w = img.shape[-2], img.shape[-1]
    if jitter is None:
        jitter = min(h,w)//2
    fastai_image = vision.Image(img.squeeze())

    # pad
    fastai_image._flow = gpu_affine_grid(fastai_image.shape, device)
    vision.transform.pad()(fastai_image, jitter)

    # jitter
    first_jitter = int((jitter*(2/3)))
    vision.transform.crop_pad()(fastai_image,
                                (h+first_jitter,w+first_jitter),
                                row_pct=np.random.rand(), col_pct=np.random.rand())

    # scale
    percent = scale * 100 # scale up to integer to avoid float repr errors
    scale_factors = [(100 - percent + percent/5. * i)/100 for i in range(11)]
    rand_scale = scale_factors[int(np.random.rand()*len(scale_factors))]
    fastai_image._flow = gpu_affine_grid(fastai_image.shape, device)
    vision.transform.zoom()(fastai_image, rand_scale)

    # rotate
    rotate_factors = list(range(-degrees, degrees+1)) + degrees//2 * [0]
    rand_rotate = rotate_factors[int(np.random.rand()*len(rotate_factors))]
    fastai_image._flow = gpu_affine_grid(fastai_image.shape, device)
    vision.transform.rotate()(fastai_image, rand_rotate)

    # jitter
    vision.transform.crop_pad()(fastai_image, (h,w), row_pct=np.random.rand(), col_pct=np.random.rand())

    return fastai_image.data[None,:]


# def lucid_transforms_vol(vol, jitter=None, scale=.5, degrees=45):
#
#     d,h,w = vol.shape[-3], vol.shape[-2], vol.shape[-1]
#     if jitter is None:
#         jitter = min(d,h,w)//2
#     fastai_vol = vision.Image(vol.squeeze())
#
#     # pad
#     fastai_vol._flow = gpu_affine_grid(fastai_vol.shape)  # (30, 150, 224)
#     vision.transform.pad()(fastai_vol, jitter)
#
#     # jitter
#     first_jitter = int((jitter*(2/3)))
#     vision.transform.crop_pad()(fastai_vol,
#                                 (h+first_jitter,w+first_jitter),
#                                 row_pct=np.random.rand(), col_pct=np.random.rand())
#
#     # scale
#     percent = scale * 100 # scale up to integer to avoid float repr errors
#     scale_factors = [(100 - percent + percent/5. * i)/100 for i in range(11)]
#     rand_scale = scale_factors[int(np.random.rand()*len(scale_factors))]
#     fastai_vol._flow = gpu_affine_grid(fastai_vol.shape)
#     vision.transform.zoom()(fastai_vol, rand_scale)
#
#     # rotate
#     rotate_factors = list(range(-degrees, degrees+1)) + degrees//2 * [0]
#     rand_rotate = rotate_factors[int(np.random.rand()*len(rotate_factors))]
#     fastai_vol._flow = gpu_affine_grid(fastai_vol.shape)
#     vision.transform.rotate()(fastai_vol, rand_rotate)
#
#     # jitter
#     vision.transform.crop_pad()(fastai_vol, (h,w), row_pct=np.random.rand(), col_pct=np.random.rand())
#
#     return fastai_vol.data[None,:]


def tensor_stats(t, label=""):
    if len(label) > 0: label += " "
    return("%smean:%.2f std:%.2f max:%.2f min:%.2f" % (label, t.mean().item(),t.std().item(),t.max().item(),t.min().item()))

def cossim(act0, act1, cosim_weight=0):
    dot = (act0 * act1).sum()
    mag0 = act0.pow(2).sum().sqrt()
    mag1 = act1.pow(2).sum().sqrt()
    cossim = cosim_weight*dot/(mag0*mag1)
    return cossim

# def visualize_feature(model, layer, feature, start_image=None, last_hook_out=None,
#                       size=200, steps=500, lr=0.004, weight_decay=0.1, grad_clip=1,
#                       debug=False, frames=10, show=True):
#     h,w = size if type(size) is tuple else (size,size)
#     if start_image is not None:
#         fastai_image = vision.Image(start_image.squeeze())
#         fastai_image._flow = gpu_affine_grid((3,h,w)) # resize
#         img_buf = fastai_image.data[None,:]
#         img_buf = normalize(img_buf)
#         img_buf = rgb_to_lucid_colorspace(img_buf)
#         img_buf = rgb_to_fft(h, w, img_buf)
#     else:
#         img_buf = init_fft_buf(h, w)
#     img_buf.requires_grad_()
#     opt = torch.optim.AdamW([img_buf], lr=lr, weight_decay=weight_decay)
#
#     hook_out = None
#     def callback(m, i, o):
#         nonlocal hook_out
#         hook_out = o
#     hook = layer.register_forward_hook(callback)
#
#     for i in range(1,steps+1):
#         opt.zero_grad()
#
#         img = fft_to_rgb(h, w, img_buf)
#         img = lucid_colorspace_to_rgb(img)
#         stats = tensor_stats(img)
#         img = torch.sigmoid(img)
#         img = normalize(img)
#         img = lucid_transforms(img)
#         model(img.cuda(device))
#         if feature is None:
#             loss = -1 * hook_out[0].pow(2).mean()
#         else:
#             loss = -1 * hook_out[0][feature].mean()
#         if last_hook_out is not None:
#             simularity = cossim(hook_out[0], last_hook_out)
#             loss = loss + loss * simularity
#
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(img_buf,grad_clip)
#         opt.step()
#
#         if debug and (i)%(int(steps/frames))==0:
#             # clear_output(wait=True)
#             label = f"step: {i} loss: {loss:.2f} stats:{stats}"
#             show_rgb(image_buf_to_rgb(h, w, img_buf),
#                      label=label)
#
#     hook.remove()
#
#     retval = image_buf_to_rgb(h, w, img_buf)
#     if show:
#         if not debug: show_rgb(retval)
#     return retval, hook_out[0].clone().detach()