from skimage.exposure import rescale_intensity
import numpy as np
import glob
import os
import cv2

def GBF(ori, guide, sigma_s, sigma_r, target_dir, fname):
    output = []
    r = 3*sigma_s
    iH, iW, ch = ori.shape
    s_grid = np.array([[((i**2+j**2)/(2.0*(sigma_s**2))) for i in range(-r, r+1)] for j in range(-r, r+1)])
    s_ker = np.exp(-s_grid)
    ori = cv2.copyMakeBorder(ori, r, r, r, r, cv2.BORDER_REPLICATE)
    guide = cv2.copyMakeBorder(guide, r, r, r, r, cv2.BORDER_REPLICATE)
    
    if len(guide.shape)==2:
        for y in range(r, iH+r):
            for x in range(r, iW+r):
                ori_r = ori[y-r:y+r+1, x-r:x+r+1, 2]
                ori_g = ori[y-r:y+r+1, x-r:x+r+1, 1]
                ori_b = ori[y-r:y+r+1, x-r:x+r+1, 0]
                grid_guide = guide[y-r:y+r+1, x-r:x+r+1]
                minus_guide = np.ones((2*r+1, 2*r+1))*guide[y][x]
                element_guide = np.exp(np.power(grid_guide - minus_guide, 2)/(-2*(sigma_r**2)))
                combined_ker = s_ker * element_guide
                inv_ori_r = np.flip(ori_r)
                inv_ori_g = np.flip(ori_g)
                inv_ori_b = np.flip(ori_b)
                output_r = np.sum(combined_ker * inv_ori_r)/np.sum(combined_ker)
                output_g = np.sum(combined_ker * inv_ori_g)/np.sum(combined_ker)
                output_b = np.sum(combined_ker * inv_ori_b)/np.sum(combined_ker)
                output.append([output_b, output_g, output_r])
    else:
        for y in np.arange(r, iH+r):
            for x in np.arange(r, iW+r):
                ori_r = ori[y-r:y+r+1, x-r:x+r+1, 2]
                ori_g = ori[y-r:y+r+1, x-r:x+r+1, 1]
                ori_b = ori[y-r:y+r+1, x-r:x+r+1, 0]
                minus_r = np.ones((2*r+1, 2*r+1))*ori[y][x][2]
                minus_g = np.ones((2*r+1, 2*r+1))*ori[y][x][1]
                minus_b = np.ones((2*r+1, 2*r+1))*ori[y][x][0]
                element_r = np.exp(np.power(ori_r - minus_r, 2)/(-2*(sigma_r**2)))
                element_g = np.exp(np.power(ori_g - minus_g, 2)/(-2*(sigma_r**2)))
                element_b = np.exp(np.power(ori_b - minus_b, 2)/(-2*(sigma_r**2)))
                combined_ker = s_ker * element_r * element_g * element_b
                inv_ori_r = np.flip(ori_r)
                inv_ori_g = np.flip(ori_g)
                inv_ori_b = np.flip(ori_b)
                output_r = np.sum(combined_ker * inv_ori_r)/np.sum(combined_ker)
                output_g = np.sum(combined_ker * inv_ori_g)/np.sum(combined_ker)
                output_b = np.sum(combined_ker * inv_ori_b)/np.sum(combined_ker)
                output.append([output_b, output_g, output_r])
        
    output = np.array(output, dtype='float')
    #image = rescale_intensity(output.reshape(iH, iW, ch), in_range=(0, 255))
    #image = (image * 255).astype("uint8")
    #cv2.imshow("test", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(target_dir, fname), output.reshape(iH, iW, ch)*255.0)


sigma_s = [1, 2, 3]
sigma_r = [0.05, 0.1, 0.2]
rt_dict = {"0a_img_66":"0a.png", "0b_img_66":"0b.png", "0c_img_66":"0c.png"}
#rt_dict = {"0b_img_66":"0b.png", "0c_img_66":"0c.png"}

for rt_dir, src in rt_dict.items():
    ori = cv2.imread(src)
    ori = ori.astype('float')/255.0
    for s in sigma_s:
        for r in sigma_r:
            target_dir = os.path.join(rt_dir,"s_"+str(s)+"r_"+str(r))
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            for path in glob.iglob(os.path.join(rt_dir,"*.png")):
                guide = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                guide = guide.astype('float')/255.0
                GBF(ori, guide, s, r, target_dir, os.path.basename(path))
            GBF(ori, ori, s, r, target_dir, src)
