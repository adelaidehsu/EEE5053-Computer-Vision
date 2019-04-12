import numpy as np
import cv2
#from utils import get_four_points

# u, v are N-by-2 matrices, representing N corresponding points for v = T(u)
# this function should return a 3-by-3 homography matrix
def solve_homography(u, v):
    N = u.shape[0]
    if v.shape[0] is not N:
        print('u and v should have the same size')
        return None
    if N < 4:
        print('At least 4 points should be given')
	# if you take solution 2:
    A = np.zeros((2*N, 9))
    b = np.zeros((2*N, 1))
    H = np.zeros((3, 3))
    # TODO: compute H from A and b
    for i in range(N):
        A[i*2,:] = [u[i][0], u[i][1], 1, 0, 0, 0, -v[i][0]*u[i][0], -v[i][0]*u[i][1], -v[i][0]]
        A[i*2+1,:] = [0, 0, 0, u[i][0], u[i][1], 1, -v[i][1]*u[i][0], -v[i][1]*u[i][1], -v[i][1]]

    [U,S,V] = np.linalg.svd(A)
    m = V[-1,:]
    H = np.reshape(m,(3,3))
    return H

# corners are 4-by-2 arrays, representing the four image corner (x, y) pairs
def transform(img, canvas, corners):
    h, w, ch = img.shape
    cv2.fillConvexPoly(canvas, corners.astype(int), 0, 16)
    # TODO: some magic
    canvas += img.astype(canvas.dtype)

def mapping(img, canvas, H):
    h, w, ch = img.shape
    hc, wc, chc = canvas.shape
    final = np.zeros((hc, wc, chc))
    for i in range(w):
        for j in range(h):
            point = np.array([[i], [j], [1]])
            tmp = np.dot(H, point)
            x = int(tmp[0] / tmp[2])
            y = int(tmp[1] / tmp[2])
            if x > 0 and y > 0 and x < wc and y < hc:
                final[y, x, :] = img[j, i, :]                             
    return final

def rev_mapping(dst_size, img, H_rev):
    _, _, ch = img.shape
    h, w = dst_size[0], dst_size[1]
    final = np.zeros((h, w, ch))
    for i in range(w):
        for j in range(h):
            point = np.array([[i], [j], [1]])
            tmp = np.dot(H_rev, point)
            x = tmp[0] / tmp[2]
            y = tmp[1] / tmp[2]
            if (x-int(x) != 0) or (y-int(y) != 0):
                x = np.asarray(x)
                y = np.asarray(y)
                x0 = np.floor(x).astype(int)
                y0 = np.floor(y).astype(int)
                y1 = y0+1
                x1 = x0+1
                x0 = np.clip(x0, 0, img.shape[1]-1)
                x1 = np.clip(x1, 0, img.shape[1]-1)
                y0 = np.clip(y0, 0, img.shape[0]-1)
                y1 = np.clip(y1, 0, img.shape[0]-1)
                '''
                Ia = img[ y0, x0, : ]
                Ib = img[ y1, x0, : ]
                Ic = img[ y0, x1, : ]
                Id = img[ y1, x1, : ]
                wa = (x1-x) * (y1-y)
                wb = (x1-x) * (y-y0)
                wc = (x-x0) * (y1-y)
                wd = (x-x0) * (y-y0)
                final[j, i, :] = wa*Ia + wb*Ib + wc*Ic + wd*Id
                '''
                #method2
                d1 = x-x0
                d3 = y-y0
                tmp1 = (1-d1)*img[y0, x0, :] + d1*img[y0, x1, :]
                tmp2 = (1-d1)*img[y1, x0, :] + d1*img[y1, x1, :]
                final[j, i, :] = d3*tmp2 + (1-d3)*tmp1           
            else:
                final[j, i, :] = img[y, x, :]
    return final

def main():
    # Part 1
    canvas = cv2.imread('./input/times_square.jpg')
    img1 = cv2.imread('./input/1.jpg')
    img2 = cv2.imread('./input/2.jpg')
    img3 = cv2.imread('./input/3.jpg')
    img4 = cv2.imread('./input/4.jpg')
    img5 = cv2.imread('./input/5.jpg')
    size1 = img1.shape
    size2 = img2.shape
    size3 = img3.shape
    size4 = img4.shape
    size5 = img5.shape

    src, corner, imgs = [], [], []
    imgs.append(img1)
    imgs.append(img2)
    imgs.append(img3)
    imgs.append(img4)
    imgs.append(img5)
    src.append(np.array([[0,0], [0, size1[0]-1], [size1[1]-1, size1[0]-1], [size1[1]-1, 0]]))
    src.append(np.array([[0,0], [0, size2[0]-1], [size2[1]-1, size2[0]-1], [size2[1]-1, 0]]))
    src.append(np.array([[0,0], [0, size3[0]-1], [size3[1]-1, size3[0]-1], [size3[1]-1, 0]]))
    src.append(np.array([[0,0], [0, size4[0]-1], [size4[1]-1, size4[0]-1], [size4[1]-1, 0]]))
    src.append(np.array([[0,0], [0, size5[0]-1], [size5[1]-1, size5[0]-1], [size5[1]-1, 0]]))
    corner.append(np.array([[818, 352], [818, 407], [885, 408], [884, 352]]))
    corner.append(np.array([[311, 14], [157, 152], [278, 315], [402, 150]]))
    corner.append(np.array([[364, 674], [279, 864], [369, 885], [430, 725]]))
    corner.append(np.array([[808, 495], [802, 609], [896, 609], [892, 495]]))
    corner.append(np.array([[1024, 608], [1032, 664], [1134, 651], [1118, 593]]))

    H = []
    for i in range(5):
        H.append(solve_homography(src[i], corner[i]))
    for i in range(5):
        tmp = mapping(imgs[i], canvas, H[i])
        transform(tmp, canvas, corner[i])
    cv2.imwrite('part1.png', canvas)
    
    # Part 2
    img = cv2.imread('./input/screen.jpg')
    pts1 = np.float32([[1040, 371], [1102, 397], [984, 553], [1036, 602]])
    pts2 = np.float32([[0, 0], [500, 0], [0, 500], [500, 500]])
    H = solve_homography(pts2, pts1)
    output2 = rev_mapping((500,500), img, H)
    cv2.imwrite('part2.png', output2)
    
    # Part 3
    img_front = cv2.imread('./input/crosswalk_front.jpg')
    #pts1 = np.float32([[95, 157], [614, 152], [33, 241], [680, 231]])
    #pts1 = np.float32([[104, 157], [612, 152], [18, 279], [688, 270]])
    pts1 = np.float32([[33, 132], [653, 128], [25, 292], [686, 287]])
    pts2 = np.float32([[0, 0], [800, 0], [0, 600], [800, 600]])
    H = solve_homography(pts2, pts1)
    output3 = rev_mapping((600, 800), img_front, H)
    #src = get_four_points(img_front)
    #print(src)
    cv2.imwrite('part3.png', output3)
    

if __name__ == '__main__':
    main()
