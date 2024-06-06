import cv2
import os
import matplotlib.pyplot as plt

IMAGES_PATH = '/home/zhao/DataSets/ISPRS_dataset/Vaihingen/SAM_O_M/ISPRS_OBJECT_merge_1.tif'
image = cv2.imread(IMAGES_PATH,cv2.IMREAD_GRAYSCALE)

if image is None:
    print("path is error")
else:
    plt.imshow(image,cmap='gray')
    plt.title('single channel image')
    plt.axis('off')
    plt.colorbar()
    plt.show()


# rgb_image = cv2.merge([image]*3)
# cv2.imshow('RGB image',rgb_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
rgb_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
plt.imshow(rgb_image)
plt.title('rgb image')
plt.axis('off')
plt.show()