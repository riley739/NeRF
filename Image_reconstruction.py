import cv2
import math
import torch

# og = torch.tensor(cv2.imread("test1.png"))
og = cv2.imread("test1.png")
# images = torch.reshape(og,(og.size(0)*og.size(1),3))
# # A = [1.8,0.25,1.5]
# # B = [1.9,0.25,1.5]
# A = [1,1,1]
# B = [1,1,1]
A = 1
B = 1
d = 1
# depths = torch.ones(images.size(0),1)*1
# print(depths.size())
# print(images.size())

# r,g,b = images.unbind(1)


# r_part = math.e**(-B[0]*depths) + A[0]*(1 - math.e**(-B[0]*depths))

# r_part = torch.flatten(r_part)

# r_hazy = r * r_part

# g_part = math.e**(-B[1]*depths) + A[1]*(1 - math.e**(-B[1]*depths))
# g_part = torch.flatten(g_part)
# g_hazy = g * g_part

# b_part = math.e**(-B[2]*depths) + A[2]*(1 - math.e**(-B[2]*depths))
# b_part = torch.flatten(b_part)
# b_hazy = b * b_part
# size = r_hazy.size(dim=0)
# r_hazy = torch.reshape(r_hazy,(size,1))
# g_hazy = torch.reshape(g_hazy,(size,1))
# b_hazy = torch.reshape(b_hazy,(size,1))
# hazy = torch.cat((r_hazy,b_hazy,g_hazy), dim=1)

# hazy = torch.reshape(hazy, (og.size(0), og.size(1), 3))

# hazy = hazy.numpy()
hazy = og*math.e**(-B*d) + A*(1 - math.e**(-B*d))

cv2.imwrite("yeww.png",hazy)

