import torch
import math
import cv2 

DEFAULT_A = [0.1,0.3,0.2]
DEFAULT_B = [1,1,1]
    
def light_model(images, depths, A = None, B = None):
    # if A is None:
    #     A = DEFAULT_A
    
    # if B is None:
    #     B = DEFAULT_B
    # # print(depths)
    r,g,b = images.unbind(1)

    # r = r*255
    # g = g*255
    # b = b*255
    
    image = images.cpu().detach().numpy()
    depthMap = depths.cpu().detach().numpy()
    image = np.reshape(image,(depthMap.size,3))
    depthMap = np.reshape(depthMap,(depthMap.size))
    
    
    coeff_r, coeff_b, coeff_g = find_atm_light(image,depthMap)
    
    print(coeff_r)
    # r_hazy = r * torch.flatten(math.e**(-coeff_r[3]*depths))  + torch.flatten( coeff_r[0]*(1 - math.e**(-coeff_r[1]*depths)))
    r_hazy = ( r - torch.flatten( coeff_r[0]*(1 - math.e**(-coeff_r[1]*depths))) )/torch.flatten(math.e**(-coeff_r[3]*depths))

    # g_hazy = g * torch.flatten(math.e**(-coeff_g[3]*depths))  + torch.flatten( coeff_g[0]*(1 - math.e**(-coeff_g[1]*depths)))
    g_hazy = ( g - torch.flatten( coeff_g[0]*(1 - math.e**(-coeff_g[1]*depths))) )/torch.flatten(math.e**(-coeff_g[3]*depths))
    
    # b_hazy = b * torch.flatten(math.e**(-coeff_b[3]*depths))  + torch.flatten( coeff_b[0]*(1 - math.e**(-coeff_b[1]*depths)))
    b_hazy = ( b - torch.flatten( coeff_b[0]*(1 - math.e**(-coeff_b[1]*depths))) )/torch.flatten(math.e**(-coeff_b[3]*depths))


    size = r_hazy.size(dim=0)
    r_hazy = torch.reshape(r_hazy,(size,1))
    g_hazy = torch.reshape(g_hazy,(size,1))
    b_hazy = torch.reshape(b_hazy,(size,1))

    # hazy = torch.cat((r_hazy/255,g_hazy/255,b_hazy/255), dim=1)
    hazy = torch.cat((r_hazy,g_hazy,b_hazy), dim=1)
    
    
    torch.clamp_(hazy, min=0.0, max=1.0)

    # image = torch.reshape(images,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # image = image.cpu().detach().numpy()
    
    
    # cv2.imwrite("og.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR)*255)
    
    
    # hazy1 = torch.reshape(hazy,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # hazy1 = hazy1.cpu().detach().numpy()
    
    # cv2.imwrite("hazy255.png",cv2.cvtColor(hazy1, cv2.COLOR_RGB2BGR)*255)
    
    return hazy


import sys
import numpy as np
import scipy as sp
import math
from PIL import Image

def find_backscatter_estimation_points(img, depths, num_bins=10, fraction=0.01, max_vals=20, min_depth_percent=0.0):
    z_max, z_min = np.max(depths), np.min(depths)
    min_depth = z_min + (min_depth_percent * (z_max - z_min))
    z_ranges = np.linspace(z_min, z_max, num_bins + 1)
    img_norms = np.mean(img, axis=1)
    points_r = []
    points_g = []
    points_b = []
    for i in range(len(z_ranges) - 1):
        a, b = z_ranges[i], z_ranges[i+1]
        locs = np.where(np.logical_and(depths > min_depth, np.logical_and(depths >= a, depths <= b)))
        norms_in_range, px_in_range, depths_in_range = img_norms[locs], img[locs], depths[locs]
        arr = sorted(zip(norms_in_range, px_in_range, depths_in_range), key=lambda x: x[0])
        points = arr[:min(math.ceil(fraction * len(arr)), max_vals)]
        points_r.extend([(z, p[0]) for n, p, z in points])
        points_g.extend([(z, p[1]) for n, p, z in points])
        points_b.extend([(z, p[2]) for n, p, z in points])
    return np.array(points_r), np.array(points_g), np.array(points_b)

def find_backscatter_values(B_pts, depths, restarts=10, max_mean_loss_fraction=0.1):
    B_vals, B_depths = B_pts[:, 1], B_pts[:, 0]
    z_max, z_min = np.max(depths), np.min(depths)
    max_mean_loss = max_mean_loss_fraction * (z_max - z_min)
    coefs = None
    best_loss = np.inf
    def estimate(depths, B_inf, beta_B, J_prime, beta_D_prime):
        val = (B_inf * (1 - np.exp(-1 * beta_B * depths))) + (J_prime * np.exp(-1 * beta_D_prime * depths))
        return val
    def loss(B_inf, beta_B, J_prime, beta_D_prime):
        val = np.mean(np.abs(B_vals - estimate(B_depths, B_inf, beta_B, J_prime, beta_D_prime)))
        return val
    bounds_lower = [0,0,0,0]
    bounds_upper = [1,5,1,5]
    for _ in range(restarts):
        try:
            optp, pcov = sp.optimize.curve_fit(
                f=estimate,
                xdata=B_depths,
                ydata=B_vals,
                p0=np.random.random(4) * bounds_upper,
                bounds=(bounds_lower, bounds_upper),
            )
            l = loss(*optp)
            if l < best_loss:
                best_loss = l
                coefs = optp
        except RuntimeError as re:
            print(re, file=sys.stderr)
    # if best_loss > max_mean_loss:
    #     print('Warning: could not find accurate reconstruction. Switching to linear model.', flush=True)
    #     slope, intercept, r_value, p_value, std_err = sp.stats.linregress(B_depths, B_vals)
    #     BD = (slope * depths) + intercept
    #     return BD, np.array([slope, intercept])
    return estimate(depths, *coefs), coefs

def find_atm_light(image,depths):
    ptsR, ptsG, ptsB = find_backscatter_estimation_points(image, depths, fraction=0.01, min_depth_percent=0)


    Br, coefsR = find_backscatter_values(ptsR, depths, restarts=25)
    Bg, coefsG = find_backscatter_values(ptsG, depths, restarts=25)
    Bb, coefsB = find_backscatter_values(ptsB, depths, restarts=25)
    

    return coefsR, coefsG, coefsB

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32),mode="RGB").save(f, 'TIFF')