import torch
import math
import cv2 
import numpy as np 

DEFAULT_A = [0.36108535471698466, 0.3015346593573295, 0.28175415155136363]
DEFAULT_B = [0.30481738290027693, 0.12575316833412786, 0.059533905466726766]
firstTime = True
     
def light_model(images, depths, og_imgs, A = None, B = None):
    if A is None:
        A = DEFAULT_A
    if B is None:
        B = DEFAULT_B
        
    r,g,b = images.unbind(1)
        
    r = r
    g = g
    b = b

    image = images.cpu().detach().numpy()
    depthMap = depths.cpu().detach().numpy()

    image = np.reshape(image,(depthMap.size,3))
    depthMap = np.reshape(depthMap,(depthMap.size))
    if og_imgs is None:
        V_R,V_G,V_B  = 0.36108535471698466, 0.3015346593573295, 0.28175415155136363
    else:
        print("Calculating")
        og_img =  og_imgs.cpu().detach().numpy()
        print(np.mean(og_img))
        og_img = np.reshape(og_img,(depthMap.size,3))
        ind  = find_furthest_points(depthMap)
        points = og_img[ind]
        
        V_R,V_G,V_B = find_veiling_light(points)
    # global firstTime
    # if ((V_R > 0.2) and (firstTime == True)):
    #     save_img(og_imgs,"ogTestImage1.png")
    #     V_L = og_imgs
    #     print(V_L[1,:])
    #     V_L[ind, :] = torch.tensor([1,0,0], dtype=torch.float32)
    #     save_img(V_L,"VLTestImage1.png")
        
    #     np.savetxt(f"logs/log_ogTestImage1",og_img, fmt="%10.5f")
    #     np.savetxt(f"logs/log_VLTestImage1",V_L, fmt="%10.5f")
    #     np.savetxt(f"logs/log_points1",ind, fmt="%10.5f")
    #     firstTime = False
        
    # if V_R < 0.07:
    #     save_img(og_imgs,"ogTestImage2.png")
    #     V_L = og_imgs
    #     V_L[ind, :] = torch.tensor([1,0,0], dtype=torch.float32)
    #     save_img(V_L,"VLTestImage2.png")
        
    #     np.savetxt(f"logs/log_ogTestImage2",og_img, fmt="%10.5f")
    #     np.savetxt(f"logs/log_VLTestImage2",V_L, fmt="%10.5f")
    #     np.savetxt(f"logs/log_points2",ind, fmt="%10.5f")
        
        
    #     exit()
    
    
    r_hazy = r * torch.flatten(math.e**(-B[0]*depths))  + torch.flatten( V_R*(1 - math.e**(-B[0]*depths)))

    g_hazy = g * torch.flatten(math.e**(-B[1]*depths))  + torch.flatten( V_G*(1 - math.e**(-B[1]*depths)))

    b_hazy = b * torch.flatten(math.e**(-B[2]*depths))  + torch.flatten( V_B*(1 - math.e**(-B[2]*depths)))


    size = r_hazy.size(dim=0)
    r_hazy = torch.reshape(r_hazy,(size,1))
    g_hazy = torch.reshape(g_hazy,(size,1))
    b_hazy = torch.reshape(b_hazy,(size,1))

    # hazy = torch.cat((r_hazy/255,g_hazy/255,b_hazy/255), dim=1)
    hazy = torch.cat((r_hazy,g_hazy,b_hazy), dim=1) 
    torch.clamp_(hazy, min=0.0, max=1.0)

    # image = torch.reshape(images,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # image = image.cpu().detach().numpy()
    
    
    # # cv2.imwrite("og.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR)*255)
    
    
    # hazy1 = torch.reshape(hazy,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # hazy1 = hazy1.cpu().detach().numpy()
    
    # cv2.imwrite("hazy255.png",cv2.cvtColor(hazy1, cv2.COLOR_RGB2BGR)*255)
    
    return hazy
def find_furthest_points(depths):
    #TODO maybe check on density values not the distance like relook at NeRF things
    # Get 5% of furthest away points 
    # TODO maybe change to 1 %
    pixels = int(depths.size*0.05)
    ind = np.argpartition(depths, pixels-1)[:pixels]

    # TODO maybe put something in here to check mean /stddev
    # max_array = depths[ind]
    
    return ind

def find_veiling_light(points):
    global count
    r,g,b  = points[:, 0], points[:,1], points[:,  2] # For RGB image

    R = np.mean(r)
    G = np.mean(g)
    B = np.mean(b)


    print(f"R is {R}, G is {G}, B is {B} ")
    return R , G , B


def save_img(img,pth):
    img = torch.reshape(img,(int(img.size(0)**(1/2)), int(img.size(0)**(1/2)),3))
    img = img.cpu().detach().numpy()*255
    
    
    cv2.imwrite(pth,cv2.cvtColor(img,cv2.COLOR_RGB2BGR))
    
# def save_img_u8(img, pth):
#     """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""

#     try:
#         img = img.cpu().detach().numpy()
#     except AttributeError:
#         pass
#     with open(pth, 'wb') as f:
#         Image.fromarray(
#             (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
#                 f, 'PNG')


# def save_img_f32(depthmap, pth):
#   """Save an image (probably a depthmap) to disk as a float32 TIFF."""
#   with open_file(pth, 'wb') as f:
#     Image.fromarray(np.nan_to_num(depthmap).astype(np.float32),mode="RGB").save(f, 'TIFF')