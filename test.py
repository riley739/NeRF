import cv2
import torch 
import numpy as np 
import math
import sys

DEFAULT_A = [0.1,0.3,0.2]
DEFAULT_B = [1,1,1]
    
def light_model(images, depths, A = None, B = None):
    
    if A is None:
        A = DEFAULT_A
    
    if B is None:
        B = DEFAULT_B

    r,g,b = images.unbind(1)

    r = r
    g = g
    b = b

    r_hazy = r * torch.flatten(math.e**(-B[0]*depths))  + torch.flatten( A[0]*(1 - math.e**(-B[0]*depths)))


    g_hazy = g * torch.flatten(math.e**(-B[1]*depths))  + torch.flatten( A[1]*(1 - math.e**(-B[1]*depths)))

    b_hazy = b * torch.flatten(math.e**(-B[2]*depths))  + torch.flatten( A[2]*(1 - math.e**(-B[2]*depths)))


    size = r_hazy.size(dim=0)
    r_hazy = torch.reshape(r_hazy,(size,1))
    g_hazy = torch.reshape(g_hazy,(size,1))
    b_hazy = torch.reshape(b_hazy,(size,1))

    hazy = torch.cat((r_hazy,g_hazy,b_hazy), dim=1)
    
    # torch.clamp_(hazy, min=0.0, max=1.0)

    # image = torch.reshape(images,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # image = image.cpu().detach().numpy()
    
    
    # # cv2.imwrite("og.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR)*255)
    
    
    # hazy1 = torch.reshape(hazy,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # hazy1 = hazy1.cpu().detach().numpy()
    
    # cv2.imwrite("hazy255.png",cv2.cvtColor(hazy1, cv2.COLOR_RGB2BGR)*255)
    
    return hazy

def hist_eq(image):

    img_y_cr_cb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_y_cr_cb)

    # Applying equalize Hist operation on Y channel.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y_eq = clahe.apply(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))
    img_rgb_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2RGB)
    
    return img_rgb_eq
    

def light_model_reverse(images, depths, A, B):
    

    r,g,b = images.unbind(1)

    r_hazy = r
    g_hazy = g
    b_hazy = b

    # r_hazy = r * torch.flatten(math.e**(-B[0]*depths))  + torch.flatten( A[0]*(1 - math.e**(-B[0]*depths)))

    r = (r_hazy - torch.flatten( A[0]*(1 - math.e**(-B[0]*depths))) ) / torch.flatten(math.e**(-B[0]*depths))

    # g_hazy = g * torch.flatten(math.e**(-B[1]*depths))  + torch.flatten( A[1]*(1 - math.e**(-B[1]*depths)))

    g = (g_hazy - torch.flatten( A[1]*(1 - math.e**(-B[1]*depths))) ) / torch.flatten(math.e**(-B[1]*depths))
    
    # b_hazy = b * torch.flatten(math.e**(-B[2]*depths))  + torch.flatten( A[2]*(1 - math.e**(-B[2]*depths)))

    b = (b_hazy - torch.flatten( A[2]*(1 - math.e**(-B[2]*depths))) ) / torch.flatten(math.e**(-B[2]*depths))

    size = r.size(dim=0)
    r = torch.reshape(r,(size,1))
    g = torch.reshape(g,(size,1))
    b = torch.reshape(b,(size,1))

    hazy = torch.cat((r,g,b), dim=1)
    

    # image = torch.reshape(images,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # image = image.cpu().detach().numpy()
    
    
    # # cv2.imwrite("og.png",cv2.cvtColor(image, cv2.COLOR_RGB2BGR)*255)
    
    
    # hazy1 = torch.reshape(hazy,(int(images.size(0)**(1/2)), int(images.size(0)**(1/2)),3))
    # hazy1 = hazy1.cpu().detach().numpy()
    
    # cv2.imwrite("hazy255.png",cv2.cvtColor(hazy1, cv2.COLOR_RGB2BGR)*255)
    
    return hazy.to_numpy()


# def gray_world_assumption(image):
#     nimg = nimg.transpose(2, 0, 1).astype(np.uint32)
#     mu_g = np.average(nimg[1])
#     nimg[0] = np.minimum(nimg[0]*(mu_g/np.average(nimg[0])),255)
#     nimg[2] = np.minimum(nimg[2]*(mu_g/np.average(nimg[2])),255)
#     return  nimg.transpose(1, 2, 0).astype(np.uint8)


def calculate_average_color(image):
    """Calculate the average color of an image."""
    height, width, _ = image.shape
    return np.mean(image.reshape(height * width, -1), axis=0)

def grey_world_assumption(image, hist):
    """Check if an image satisfies the Grey World assumption."""
    # grey_value = np.array([128, 128, 128])  # Neutral grey value in RGB (128, 128, 128)
    average_color = calculate_average_color(image)
    average_color1 = calculate_average_color(hist)
    distance_to_grey = np.linalg.norm(average_color - average_color1)
    return distance_to_grey


def compare_images(image1, image2):
    distance = 0
    height,width, _ = image1.shape
    # for i,row in enumerate(image1):
    #     for j,pixel in enumerate(row):
    #         distance += pixel - image2[i][j]
    distance = np.absolute(image1 - image2)
    rgb_dist = np.sum(distance,axis=(0,1))
    # print(rgb_dist)
    return sum(rgb_dist)/ (height * width* _)

if __name__ == "__main__":
    og_image = cv2.cvtColor(cv2.imread('image.png'),cv2.COLOR_BGR2RGB)

    hist = hist_eq(og_image)
    cv2.imwrite("test_hist.png",cv2.cvtColor(hist,cv2.COLOR_RGB2BGR))
    height,width, _ = og_image.shape
    image = og_image.reshape(og_image.shape[0]*og_image.shape[1],3)
    hist = hist.reshape(height*width,3)

    depths = torch.from_numpy(np.ones(image.shape[0]))
    image = torch.from_numpy(image)
    hist = torch.from_numpy(hist)
    
    images = []
    distances = []
    count = 0 
    A_og = [0,0,0]
    B_og = [0,0,0]
    previous_distance = sys.maxsize 


    for i in range(10000):
        A = [ min(1,max(0,i + np.random.uniform(-0.5,0.5))) for i in A_og]
        # B_val = np.random.uniform(-0.1,0.1)
        B = [ min(1,max(0,i + np.random.uniform(-0.5,0.5))) for i in B_og]

        # print(A)
        clear = light_model(hist,depths,A,B)
        clear = clear.numpy()
        clear = clear.reshape(height,width,3)
        # distance = grey_world_assumption(clear,og_image)
        distance = compare_images(clear,og_image)
        print(f"Current Count {count}, Distance {distance}")

        if distance < 25:
            images.append(clear)
            distances.append(distance)
            count += 1
            print(f"Current Count {count}, Distance {distance}, A {A} , B {B}")


        if distance <= previous_distance: 
            B_og = B
            A_og = A

        if count > 10:
            break

    for img,name in zip(images,distances):
        print(name)
        print(img.shape )
        cv2.imwrite(f"Images/Image{name}.png",cv2.cvtColor(og_image,cv2.COLOR_RGB2BGR)) 



# def dark_channel(image):
#     %Estimating Dark Channel Prior
# %This is the function that is created to give us J_DARK which is the dark channel for the hazy image

# [r,c,m] = size(hazy_image);   %Size of hazy image
# P_S = 15;    %Patch Size 15X15
# p = floor((P_S-1)/2);   %For padding
# hazy_image_padded= padarray(hazy_image,[p,p],'replicate','both'); 

# J_DARK = zeros(r,c);    %Size of dark channel image - initializing

# for i=1:r
#     for j=1:c
#         local_patch = hazy_image_padded(i:i+P_S-1,j:j+P_S-1,1:m); %Setting a patch
#         for k=1:m
#             y=local_patch(:,:,k);
#             minimum_c(1,k) = min(min(y));  %Minimum in the patch for each channel
#         end
#         J_DARK(i,j)=min(minimum_c); %Minimum of the whole patch
