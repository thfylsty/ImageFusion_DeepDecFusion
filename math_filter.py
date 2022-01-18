# import cv2
# import numpy as np
# from pywt import dwt2, idwt2
#
# # 读取灰度图
# img = cv2.imread('lena.jpg', 0)
#
# # 对img进行haar小波变换：
# cA, (cH, cV, cD) = dwt2(img, 'sym3')
#
# # # 小波变换之后，低频分量对应的图像：
# # cv2.imwrite('lena.png', np.uint8(cA ))
# # # 小波变换之后，水平方向高频分量对应的图像：
# # cv2.imwrite('lena_h.png', np.uint8(cH))
# # # 小波变换之后，垂直平方向高频分量对应的图像：
# # cv2.imwrite('lena_v.png', np.uint8(cV ))
# # # 小波变换之后，对角线方向高频分量对应的图像：
# # cv2.imwrite('lena_d.png', np.uint8(cD ))
#
# # 小波变换之后，低频分量对应的图像：
# cv2.imwrite('lena1.png',cA /2)
# # 小波变换之后，水平方向高频分量对应的图像：
# cv2.imwrite('lena_h1.png',cH*50)
# # 小波变换之后，垂直平方向高频分量对应的图像：
# cv2.imwrite('lena_v1.png', cV*50 )
# # 小波变换之后，对角线方向高频分量对应的图像：
# cv2.imwrite('lena_d1.png', cD *50 )
#
# # 根据小波系数重构回去的图像
# rimg = idwt2((cA, (cH, cV, cD)), 'haar')
# cv2.imwrite('rimg.png', np.uint8(rimg))
import time

import cv2
import numpy as np
img = cv2.imread("lena.jpg",0)
s_time = time.time()
dst = cv2.Laplacian(img,cv2.CV_16S,ksize=3)
e_time = time.time() - s_time
print(e_time)
dst = cv2.convertScaleAbs(dst)
imglpls = img-dst

gs = cv2.GaussianBlur(img,(3,3),0)

# cv2.imshow("imglpls",imglpls)
# cv2.imshow("gs",gs)
# cv2.imshow("imglplsimglpls",imglpls-gs)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite('lena_lpls.png', dst*1.5 )