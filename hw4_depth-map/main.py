import numpy as np
import cv2
from cv2.ximgproc import jointBilateralFilter, weightedMedianFilter, guidedFilter
import time

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.uint8)
    dspMap_l = np.zeros((h, w), dtype=np.uint8)
    dspMap_r = np.zeros((h, w), dtype=np.uint8)
    SDI_l = np.zeros((h, w, max_disp), dtype=np.uint8)
    SDI_r = np.zeros((h, w, max_disp), dtype=np.uint8)

    Ilg = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Irg = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    # >>> Cost computation
    tic = time.time()
    # TODO: Compute matching cost from Il and Ir
    for d in range(max_disp):
        distance = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                if x-d >= 0:
                    #ref_r0 = Ir[y, x-d, 0]
                    #ref_r1 = Ir[y, x-d, 1]
                    #ref_r2 = Ir[y, x-d, 2]
                    ref_r = Irg[y, x-d]
                else:
                    #ref_r0 = Ir[y, x, 0]
                    #ref_r1 = Ir[y, x, 1]
                    #ref_r2 = Ir[y, x, 2]
                    ref_r = Irg[y, x]
                #distance[y, x] = (Il[y, x, 0] - ref_r0)**2 + (Il[y, x, 1] - ref_r1)**2 + (Il[y, x, 2] - ref_r2)**2
                distance[y, x] = (Ilg[y, x] - ref_r)**2
        SDI_l[:, :, d] = distance
    
    for d in range(max_disp):
        distance = np.zeros((h, w))
        for y in range(h):
            for x in range(w):
                if x+d < w:
                    #ref_l0 = Il[y, x+d, 0]
                    #ref_l1 = Il[y, x+d, 1]
                    #ref_l2 = Il[y, x+d, 2]
                    ref_l = Ilg[y, x+d]
                else:
                    #ref_l0 = Il[y, x, 0]
                    #ref_l1 = Il[y, x, 1]
                    #ref_l2 = Il[y, x, 2]
                    ref_l = Ilg[y, x]
                #distance[y, x] = (Ir[y, x, 0] - ref_l0)**2 + (Ir[y, x, 1] - ref_l1)**2 + (Ir[y, x, 2] - ref_l2)**2
                distance[y, x] = (Irg[y, x] - ref_l)**2
        SDI_r[:, :, d] = distance
    toc = time.time()
    print('* Elapsed time (cost computation): %f sec.' % (toc - tic))
    #for d in range(max_disp):
    #    cv2.imwrite('sub%s.jpg'%(d), SDI_l[:, :, d])
    #exit()

    # >>> Cost aggregation
    agg_SDI_l = np.zeros((h, w, max_disp), dtype=np.uint8)
    agg_SDI_r = np.zeros((h, w, max_disp), dtype=np.uint8)
    radius = 9
    eps = 0.01**2
    tic = time.time()
    # TODO: Refine cost by aggregate nearby costs
    for d in range(max_disp):
        #agg_SDI_l[:, :, d] = jointBilateralFilter(Il, SDI_l[:, :, d], -1, 35, 11)
        agg_SDI_l[:, :, d] = guidedFilter(Il, SDI_l[:, :, d], radius, eps)
    for d in range(max_disp):
        #agg_SDI_r[:, :, d] = jointBilateralFilter(Ir, SDI_r[:, :, d], -1, 35, 11)
        agg_SDI_r[:, :, d] = guidedFilter(Ir, SDI_r[:, :, d], radius, eps)
    toc = time.time()
    print('* Elapsed time (cost aggregation): %f sec.' % (toc - tic))
    #for d in range(max_disp):
    #    cv2.imwrite('agg%s.jpg'%(d), agg_SDI_l[:, :, d])
    #exit()

    # >>> Disparity optimization
    tic = time.time()
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    for y in range(h):
        for x in range(w):
            min_dsp = np.inf
            min_d = np.inf
            for d in range(max_disp):
                if agg_SDI_l[y, x, d] < min_dsp:
                    min_dsp = agg_SDI_l[y, x, d]
                    min_d = d
            dspMap_l[y, x] = min_d
    
    for y in range(h):
        for x in range(w):
            min_dsp = np.inf
            min_d = np.inf
            for d in range(max_disp):
                if agg_SDI_r[y, x, d] < min_dsp:
                    min_dsp = agg_SDI_r[y, x, d]
                    min_d = d
            dspMap_r[y, x] = min_d
    toc = time.time()
    print('* Elapsed time (disparity optimization): %f sec.' % (toc - tic))
    #new_arr = ((dspMap_l - dspMap_l.min()) * (1/(dspMap_l.max() - dspMap_l.min()) * 255).astype('uint8'))
    #cv2.imwrite('map_l.png', new_arr)
    #new_arr = ((dspMap_r - dspMap_r.min()) * (1/(dspMap_r.max() - dspMap_r.min()) * 255).astype('uint8'))
    #cv2.imwrite('map_r.png', new_arr)
    #exit()

    # >>> Disparity refinement
    tic = time.time()
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    #LRC
    consistency_left = np.zeros((h, w), dtype=np.uint8)
    thres = 0
    for y in range(h):
        for x in range(w):
            pixel_value_l = dspMap_l[y, x]
            if x - pixel_value_l>=0:
                pixel_value_r = dspMap_r[y, x-pixel_value_l]
            else:
                pixel_value_r = dspMap_r[y, x]

            if(np.abs(pixel_value_l - pixel_value_r) <= thres):
                consistency_left[y, x] = pixel_value_l
            else:
                consistency_left[y, x]=0
    
    #hole-filling
    Fp_l = np.zeros((h, w), dtype = np.uint8)
    Fp_r = np.zeros((h, w), dtype = np.uint8)
    filled = np.zeros((h, w), dtype = np.uint8)
    for y in range(h):
        for x in range(w):
            if consistency_left[y, x] == 0:
                dx = 1
                while x-dx >=0 and consistency_left[y, x-dx]==0:
                    dx+=1
                if x-dx < 0:
                    dxx=0
                    while x-dx+dxx < 0 or consistency_left[y, x-dx+dxx]==0:
                        dxx+=1
                    Fp_l[y, x] = consistency_left[y, x-dx+dxx]
                else:
                    Fp_l[y, x] = consistency_left[y, x-dx]
                
                dx = 1
                while x+dx < w and consistency_left[y, x+dx]==0:
                    dx+=1
                if x+dx >= w:
                    dxx=0
                    while x+dx-dxx >= w or consistency_left[y, x+dx-dxx]==0:
                        dxx+=1
                    Fp_r[y, x] = consistency_left[y, x+dx-dxx]
                else:
                    Fp_r[y, x] = consistency_left[y, x+dx]
            
            else:
                Fp_l[y, x] = consistency_left[y, x]
                Fp_r[y, x] = consistency_left[y, x]

    for y in range(h):
        for x in range(w):
            filled[y, x] = min(Fp_l[y, x], Fp_r[y, x])

    #weighted-median filtering
    labels = weightedMedianFilter(Il, filled, 15)
    toc = time.time()
    print('* Elapsed time (disparity refinement): %f sec.' % (toc - tic))
    #consistency_left = ((consistency_left - consistency_left.min()) * (1/(consistency_left.max() - consistency_left.min()) * 255).astype('uint8'))
    #cv2.imwrite('lrc.png', consistency_left)
    #filled = ((filled - filled.min()) * (1/(filled.max() - filled.min()) * 255).astype('uint8'))
    #cv2.imwrite('filled.png', filled)
    #labels = ((labels - labels.min()) * (1/(labels.max() - labels.min()) * 255).astype('uint8'))
    #cv2.imwrite('median.png', labels)
    #exit()

    return labels


def main():
    
    print('Tsukuba')
    img_left = cv2.imread('./testdata/tsukuba/im3.png')
    img_right = cv2.imread('./testdata/tsukuba/im4.png')
    max_disp = 15
    scale_factor = 16
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('tsukuba.png', np.uint8(labels * scale_factor))
    
    print('Venus')
    img_left = cv2.imread('./testdata/venus/im2.png')
    img_right = cv2.imread('./testdata/venus/im6.png')
    max_disp = 20
    scale_factor = 8
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('venus.png', np.uint8(labels * scale_factor))
    
    print('Teddy')
    img_left = cv2.imread('./testdata/teddy/im2.png')
    img_right = cv2.imread('./testdata/teddy/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('teddy.png', np.uint8(labels * scale_factor))
    
    print('Cones')
    img_left = cv2.imread('./testdata/cones/im2.png')
    img_right = cv2.imread('./testdata/cones/im6.png')
    max_disp = 60
    scale_factor = 4
    labels = computeDisp(img_left, img_right, max_disp)
    cv2.imwrite('cones.png', np.uint8(labels * scale_factor))
    

if __name__ == '__main__':
    main()
