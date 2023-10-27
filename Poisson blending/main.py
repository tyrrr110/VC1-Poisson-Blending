import cv2
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix

from align_target import align_target

#source_image: image to be cloned
#target_image: image to be cloned into
#target_mask: mask of the target image
def poisson_blend(source_image, target_image, target_mask):
    # print(source_image.shape)
    # print(target_mask.shape) # equals to H,W of source and target image
    H, W = target_image.shape[:2]
    masked_idx= np.column_stack(np.where(target_mask == 1)) # rows and cols idx where mask is 1
    mask_H = mask_W = len(masked_idx)
    # print(H, W)
    # print(mask_H, mask_W)

    A = np.zeros((mask_H, mask_W))
    b = np.zeros((mask_W, 3))
    # print(b.shape)

    row_map_to_A =  {tuple(masked_idx[row]):row for row in range(len(masked_idx))}
    for idx, ji in enumerate(masked_idx):
        j = ji[0]
        i = ji[1]
        if target_mask[j+1, i] == 1 and target_mask[j-1, i] == 1 and target_mask[j, i+1] == 1 and target_mask[j, i-1] == 1: 
            A[idx,  row_map_to_A[(j,i)]] = 4
            A[idx,  row_map_to_A[(j,i-1)]] = -1
            A[idx,  row_map_to_A[(j,i+1)]] = -1
            A[idx, row_map_to_A[(j-1,i)]] = -1
            A[idx, row_map_to_A[(j+1,i)]] = -1

            b[idx, 0]= 4*source_image[j,i,0] - source_image[j,i-1,0] - source_image[j,i+1,0] - source_image[j-1,i,0] - source_image[j+1,i,0] 
            b[idx, 1]= 4*source_image[j,i,1] - source_image[j,i-1,1] - source_image[j,i+1,1] - source_image[j-1,i,1] - source_image[j+1,i,1] 
            b[idx, 2]= 4*source_image[j,i,2] - source_image[j,i-1,2] - source_image[j,i+1,2] - source_image[j-1,i,2] - source_image[j+1,i,2] 

        elif target_mask[j+1, i] == 1 and target_mask[j-1, i] == 1:
            A[idx,  row_map_to_A[(j,i)]] = 2
            A[idx, row_map_to_A[(j-1,i)]] = -1
            A[idx, row_map_to_A[(j+1,i)]] = -1

            b[idx, 0]= 2*source_image[j,i,0] - source_image[j-1,i,0] - source_image[j+1,i,0] 
            b[idx, 1]= 2*source_image[j,i,1] - source_image[j-1,i,1] - source_image[j+1,i,1] 
            b[idx, 2]= 2*source_image[j,i,2] - source_image[j-1,i,2] - source_image[j+1,i,2]
        
        elif target_mask[j, i+1] == 1 and target_mask[j, i-1] == 1:
            A[idx,  row_map_to_A[(j,i)]] = 2
            A[idx,  row_map_to_A[(j,i-1)]] = -1
            A[idx,  row_map_to_A[(j,i+1)]] = -1
            b[idx, 0]= 2*source_image[j,i,0] - source_image[j,i-1,0] - source_image[j,i+1,0]  
            b[idx, 1]= 2*source_image[j,i,1] - source_image[j,i-1,1] - source_image[j,i+1,1]  
            b[idx, 2]= 2*source_image[j,i,2] - source_image[j,i-1,2] - source_image[j,i+1,2] 
        else:
            A[idx,  row_map_to_A[(j,i)]] = 1
            b[idx, 0]= target_image[j,i,0]  
            b[idx, 1]= target_image[j,i,1]  
            b[idx, 2]= target_image[j,i,2] 

    # print(A.shape)
    A = csr_matrix(A)
    v = np.zeros((mask_W, 3))
    v[:, 0] = spsolve(A, b[:,0])
    v[:, 1] = spsolve(A, b[:,1])
    v[:, 2] = spsolve(A, b[:,2])
    result_image = target_image.copy()
    result_image[masked_idx[:,0], masked_idx[:,1]]= np.clip(v, a_min=0, a_max=255)

    lse = [np.linalg.norm(A @ v[:, 0] - b[:, 0]), np.linalg.norm(A @ v[:, 1] - b[:, 1]), np.linalg.norm(A @ v[:, 2] - b[:, 2])]
    print(f"LSE for BGR channel respectively:\n{lse[0]:.20f}\n{lse[1]:.20f}\n{lse[2]:.20f}")

    # filename = "poisson_blending_source1_output1.jpg"
    # filename = "poisson_blending_source1_output2.jpg"
    # filename = "poisson_blending_source2_output1.jpg"
    filename = "poisson_blending_source2_output2.jpg"
    cv2.imwrite(filename, result_image)

    cv2.imshow("result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


if __name__ == '__main__':
    #read source and target images
    # source_path = 'Poisson blending/source1.jpg'
    source_path = 'Poisson blending/source2.jpg'
    target_path = 'Poisson blending/target.jpg'
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    #align target image
    im_source, mask = align_target(source_image, target_image)

    ##poisson blend
    blended_image = poisson_blend(im_source, target_image, mask)