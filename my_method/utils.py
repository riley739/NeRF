import torch
import math

DEFAULT_A = [0.36108535471698466, 0.3015346593573295, 0.28175415155136363]
DEFAULT_B = [0.30481738290027693, 0.12575316833412786, 0.059533905466726766]
     
def light_model(images, depths, A = None, B = None):
    
    if A is None:
        A = DEFAULT_A
    
    if B is None:
        B = DEFAULT_B

    r,g,b = images.unbind(1)

    r = r*255
    g = g*255
    b = b*255

    r_hazy = r * torch.flatten(math.e**(-B[0]*depths))  + torch.flatten( A[0]*(1 - math.e**(-B[0]*depths)))


    g_hazy = g * torch.flatten(math.e**(-B[1]*depths))  + torch.flatten( A[1]*(1 - math.e**(-B[1]*depths)))

    b_hazy = b * torch.flatten(math.e**(-B[2]*depths))  + torch.flatten( A[2]*(1 - math.e**(-B[2]*depths)))


    size = r_hazy.size(dim=0)
    r_hazy = torch.reshape(r_hazy,(size,1))
    g_hazy = torch.reshape(g_hazy,(size,1))
    b_hazy = torch.reshape(b_hazy,(size,1))

    hazy = torch.cat((r_hazy/255,g_hazy/255,b_hazy/255), dim=1)
    
    torch.clamp_(hazy, min=0.0, max=1.0)

    # image = torch.reshape(images,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # image = image.cpu().detach().numpy()
    
    
    # # cv2.imwrite("og.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR)*255)
    
    
    # hazy1 = torch.reshape(hazy,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # hazy1 = hazy1.cpu().detach().numpy()
    
    # cv2.imwrite("hazy255.png",cv2.cvtColor(hazy1, cv2.COLOR_RGB2BGR)*255)
    
    return hazy
