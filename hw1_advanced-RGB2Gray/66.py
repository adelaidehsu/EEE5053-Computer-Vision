import os
import numpy as np
import cv2

#generate 66 imgs
def rgb_lin_cvt(img_str, pos, img_path):
    img = cv2.imread(img_str)
    for wr in np.arange(11):
        wr = wr/10
        for wg in np.arange(round(10*(1.1-wr),1)):
            wg = wg / 10
            wb = abs(round(1 - wr - wg, 1))
            if(not os.path.exists(os.path.join(img_path, img_str[:pos]+"_"+str(wr)+"_"+str(wg)+"_"+str(wb)+img_str[pos:]))):
                b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]
                gray = wr*r + wg*g + wb*b
                cv2.imshow('test', gray)
                cv2.imwrite(os.path.join(img_path, img_str[:pos]+"_"+str(wr)+"_"+str(wg)+"_"+str(wb)+img_str[pos:]), gray)

images = ["0a.png", "0b.png", "0c.png"]
for x in images:
    pos = x.find(".")
    img_path = x[:pos]+"_img_66"
    if not os.path.exists(img_path):
        os.makedirs(img_path)
    rgb_lin_cvt(x, pos, img_path)


'''
def rgb_lin_cvt(img_str, pos, img_path, npy_path):
    img = mpimg.imread(img_str)
    print(img)
    exit()
    if not os.path.exists(os.path.join(npy_path, img_str)):
        np.save(os.path.join(npy_path, img_str), img)
    for wr in np.arange(11):
        wr = wr/10
        for wg in np.arange(round(10*(1.1-wr),1)):
            wg = wg / 10
            wb = abs(round(1 - wr - wg, 1))
            if (not os.path.exists(os.path.join(img_path, img_str[:pos]+"_"+str(wr)+"_"+str(wg)+"_"+str(wb)+img_str[pos:]))) or \
                (not os.path.exists(os.path.join(npy_path, img_str[:pos]+"_"+str(wr)+"_"+str(wg)+"_"+str(wb)+img_str[pos:]+".npy"))):
                gray = np.dot(img[...,:3], [wr, wg, wb])
                #r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
                #gray = wr*r + wg*g + wb*b
                plt.imshow(gray, cmap = 'gray')
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                plt.savefig(os.path.join(img_path, img_str[:pos]+"_"+str(wr)+"_"+str(wg)+"_"+str(wb)+img_str[pos:]), bbox_inches='tight', pad_inches=0)
                np.save(os.path.join(npy_path, img_str[:pos]+"_"+str(wr)+"_"+str(wg)+"_"+str(wb)+img_str[pos:]), gray)
                plt.clf()
                plt.cla()
                plt.close()
'''