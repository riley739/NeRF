import cv2
import math
import torch


def hist_eq(image):

    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    y_eq = cv2.equalizeHist(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2RGB)
    
    return img_rgb_eq
    
    
og = torch.tensor(cv2.cvtColor(cv2.imread("test.png"), cv2.COLOR_BGR2RGB))
images = torch.reshape(og,(og.size(0)*og.size(1),3))
A = [0.4,0.6,0.2]

B =  [0.4, 0.1, 0.8]
a = 1
beta = 1
d = 1
depths = torch.ones(images.size(0),1)*1


r,g,b = images.unbind(1)

r_hazy = r/255
g_hazy = g/255
b_hazy = b/255

# r_hazy = r * torch.flatten(math.e**(-B[0]*depths))  + torch.flatten( A[0]*(1 - math.e**(-B[0]*depths)))
print(r_hazy)
print(r_hazy - torch.flatten( A[0]*(1 - math.e**(-B[0]*depths))))
r = (r_hazy - torch.flatten( A[0]*(1 - math.e**(-B[0]*depths))) ) / torch.flatten(math.e**(-B[0]*depths))

# g_hazy = g * torch.flatten(math.e**(-B[1]*depths))  + torch.flatten( A[1]*(1 - math.e**(-B[1]*depths)))

g = (g_hazy - torch.flatten( A[1]*(1 - math.e**(-B[1]*depths))) ) / torch.flatten(math.e**(-B[1]*depths))

# b_hazy = b * torch.flatten(math.e**(-B[2]*depths))  + torch.flatten( A[2]*(1 - math.e**(-B[2]*depths)))

b = (b_hazy - torch.flatten( A[2]*(1 - math.e**(-B[2]*depths))) ) / torch.flatten(math.e**(-B[2]*depths))

size = r.size(dim=0)
r = torch.reshape(r*255,(size,1))
g = torch.reshape(g*255,(size,1))
b = torch.reshape(b*255,(size,1))

hazy = torch.cat((r,g,b), dim=1)

hazy = torch.reshape(hazy, (og.size(0), og.size(1), 3))

hazy = hazy.numpy()

cv2.imwrite("output.png",cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))


image = hist_eq(cv2.cvtColor(cv2.imread("test.png"), cv2.COLOR_BGR2RGB))
cv2.imwrite("hist.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
# # hazy = og*math.e**(-beta*d) + a*(1 - math.e**(-beta*d))
 
# hazy = (og -  a*(1 - math.e**(-beta*d)) )/math.e**(-beta*d)

# hazy = hazy.numpy()

# cv2.imwrite("output.png",cv2.cvtColor(hazy, cv2.COLOR_RGB2BGR))

