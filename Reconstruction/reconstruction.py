import numpy as np
import cv2
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

if __name__ == '__main__':
    ##read source image
    # img_src_path =  "Reconstruction/large.jpg"
    # img_src_path =  "Reconstruction/large1.jpg"
    # img_src_path =  "Reconstruction/target.jpg"
    img_src_path =  "Reconstruction/target1.jpg"
    
    img_src = cv2.imread(img_src_path)
    img_src = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

    ## implement reconstruction
    H, W = img_src.shape

    A = np.zeros((H*W, H*W))
    row_map_to_A =  {}
    idx = 0
    for j in range(H):
        for i in range(W):
            if i != 0 and j != 0 and i != W-1 and j != H-1: 
                A[idx,  j * W + i] = 4
                A[idx,  j * W + i - 1] = -1
                A[idx,  j * W + i + 1] = -1
                A[idx, (j - 1) * W + i] = -1
                A[idx, (j + 1) * W + i] = -1
            row_map_to_A[(i, j)] = idx
            idx += 1

    ## edge cases
    for i in range(1, W-1):
        idx = row_map_to_A[(i, 0)]
        A[idx, i] = 2
        A[idx, i - 1] = -1
        A[idx, i + 1] = -1

        idx = row_map_to_A[(i, H-1)]
        A[idx, (H-1) * W + i] = 2
        A[idx, (H-1) * W + (i - 1)] = -1
        A[idx, (H-1) * W + (i + 1)] = -1

    for j in range(1, H-1): 
        idx = row_map_to_A[(0, j)]
        A[idx, j * W] = 2
        A[idx, (j - 1) * W] = -1
        A[idx, (j + 1) * W] = -1
        
        idx = row_map_to_A[(W-1, j)]
        A[idx, j * W + W-1] = 2
        A[idx, (j - 1) * W + W-1] = -1
        A[idx, (j + 1) * W + W-1] = -1

    A[row_map_to_A[(0, 0)], 0] = 1
    A[row_map_to_A[(0, H-1)], (H-1) * W] = 1
    A[row_map_to_A[(W-1, 0)], W-1] = 1
    A[row_map_to_A[(W-1, H-1)], (H-1) * W + W-1] = 1

    # convert to compressed sparse row matrix for more efficient computation
    A = csr_matrix(A)


    # solve for b
    b = A @ img_src.flatten()
    # cv2.imshow("b", b.reshape([H,W]))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    ## constraints on four corner pixels
    # (0,0) (0,H-1) (W-1,0) (W-1,H-1)
    # C = [20, 20, 20, 20] 
    # C = [255, 255, 255, 255] 
    # C = [215, 215, 215, 215] 
    C = [128, 128, 128, 128] 
    # C = [20, 215, 20, 215] 
    # C = [20, 20, 215, 215] 

    b[0] = C[0]
    b[(H-1) * W] = C[1]
    b[W-1] = C[2]
    b[(H-1) * W + W-1] = C[3]

    ## solve for v
    v = spsolve(A, b.T)
    print('error is: {:.20f}'.format(np.linalg.norm(A@v - b)))

    img_tar = v.reshape([H, W]) 

    # filename = "reconstruction_result1.jpg"
    filename = "reconstruction_result2.jpg"
    # filename = "reconstruction_result3.jpg"
    # filename = "reconstruction_result4.jpg"
    cv2.imwrite(filename, img_tar)

    cv2.imshow("reconstruction", img_tar/255)# for imshow() to render the pixels 
    cv2.waitKey(0)
    cv2.destroyAllWindows()