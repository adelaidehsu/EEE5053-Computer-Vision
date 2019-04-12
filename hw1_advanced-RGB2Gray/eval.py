import numpy as np
import operator
import glob
import cv2
import os


def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def absol(imageA, imageB):
    err = np.sum(abs(imageA.astype("float") - imageB.astype("float")))
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def find_lmin(can_diff, can_vote):
    for w in can_diff:
        (wr, wg, wb) = w
        if((wr-1, wg+1, wb) in can_diff):
            if(can_diff[(wr-1, wg+1, wb)] < can_diff[w]):
                continue
        if((wr-1, wg, wb+1) in can_diff):
            if(can_diff[(wr-1, wg, wb+1)] < can_diff[w]):
                continue
        if((wr, wg-1, wb+1) in can_diff):
            if(can_diff[(wr, wg-1, wb+1)] < can_diff[w]):
                continue
        if((wr+1, wg-1, wb) in can_diff):
            if(can_diff[(wr+1, wg-1, wb)] < can_diff[w]):
                continue
        if((wr+1, wg, wb-1) in can_diff):
            if(can_diff[(wr+1, wg, wb-1)] < can_diff[w]):
                continue
        if((wr, wg+1, wb-1) in can_diff):
            if(can_diff[(wr, wg+1, wb-1)] < can_diff[w]):
                continue
        can_vote[w] = can_vote[w]+1 if w in can_vote else 1
        

sigma_s = [1, 2, 3]
sigma_r = [0.05, 0.1, 0.2]
rt_dict = {"0a_img_66":"0a.png", "0b_img_66":"0b.png", "0c_img_66":"0c.png"}

for rt_dir, rt_file in rt_dict.items():
    can_vote = {}
    for s in sigma_s:
        for r in sigma_r:
            can_diff = {}
            target_dir = os.path.join(rt_dir, "s_"+str(s)+"r_"+str(r))
            ans = cv2.imread(os.path.join(target_dir, rt_file))
            for fpath in glob.iglob(os.path.join(target_dir, rt_dir[:3]+'*')):
                can = cv2.imread(fpath)
                wr, wg, wb = os.path.basename(fpath)[3:14].split('_')
                #diff = mse(ans, can)
                diff = absol(ans, can)
                can_diff[(int(10*float(wr)), int(10*float(wg)), int(10*float(wb)))] = diff
            find_lmin(can_diff, can_vote)
    sort_can_vote = sorted(can_vote.items(), key = operator.itemgetter(1), reverse = True)
    if(len(sort_can_vote) > 3):
        top_lst = sort_can_vote[:3]
    else:
        top_lst = sort_can_vote
    #print(top_lst)
    print("For "+str(rt_file)+" , the top list is: ", top_lst)
