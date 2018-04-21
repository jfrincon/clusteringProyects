# JUAN FERNANDO RINCON CARDENO
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import math


def giveMeAngle(x1,y1,x2,y2):
    top = x1 * x2 + y1 * y2
    bottom = math.sqrt(x1**2+y1**2) * math.sqrt(x2**2 + y2**2)
    angle = math.degrees(math.acos(top/bottom))
    return angle


orig = cv2.imread('data/Flash.jpg')

orig = cv2.resize(orig, (200, 200))
#blur = cv2.blur(orig,(10,10))

img = cv2.medianBlur(orig,3)


LAB = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
#HLS = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
LAB[:,:,0] = 100




Z = LAB.reshape((-1,3))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()

#TRAIN
#X = img.reshape((img.shape[0] * img.shape[1], 3))
#print (X)
distortions = []
rangeK = range(1,10)
for k in rangeK:
    kmeanModel = KMeans(n_clusters=k).fit(Z)
    kmeanModel.fit(Z)
    distortion = sum(np.min(cdist(Z, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / Z.shape[0]
    distortions.append(distortion)

print ('Distortions:')
print (distortions)

maxAngle = 0
bestK = 2
for i in range(len(distortions)-2):
    x1 = -1
    y1 = distortions[i] - distortions[i+1]
    x2 = 1
    y2 = distortions[i+1] - distortions[i+2]
    angle = giveMeAngle(x1,y1,x2,y2)
    print ('angle of break #', i , ': ',angle)
    if angle > 168.0 :
        bestK = i + 2
        break
    if (angle > maxAngle):
        bestK = i+2
        maxAngle = angle

print ('K selected: ',bestK)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
K = bestK
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image



amountOfMM = 0
for i in range(K):
    centerP = np.uint8(center)
    centerP[i][0] = 0
    res = centerP[label.flatten()]
    res2 = res.reshape((LAB.shape))

    RGB = cv2.cvtColor(res2, cv2.COLOR_LAB2RGB)
    gray = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)

    ret,thresh = cv2.threshold(gray,50,255,0)

    thresh = cv2.medianBlur(thresh,3)
    thresh = cv2.medianBlur(thresh,3)
    thresh = cv2.medianBlur(thresh,3)


    rev = cv2.bitwise_not(thresh)
    RGBgray = cv2.cvtColor(rev, cv2.COLOR_GRAY2RGB)
    mask = cv2.bitwise_and(RGBgray,orig)

    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)



    avrg = np.sum(im2)/(im2.shape[0]*im2.shape[1]*255)
    if avrg < .5:
        print ('for cluster #',i,'this is probably the background')
        cv2.imshow('m&m'+str(i),im2)
        cv2.moveWindow('m&m'+str(i),375*(i%5),0+400*(int(i/5)))
        continue
    elif (len(contours) <= 1):
        print ('for cluster #',i,'there is: ', len(contours) - 1,'m&m, so it is probably wrong, too little or too many m&m')
    else:
        print ('for cluster #',i,'there is: ', len(contours) - 1,'m&m')
        amountOfMM+=1
    cv2.imshow('m&m'+str(i),mask)
    cv2.moveWindow('m&m'+str(i),375*(i%5),0+400*(int(i/5)))

print('I ended up finding ',amountOfMM, ' types of M&M')
cv2.imshow('res2',res2)
cv2.moveWindow('res2',700,800)
cv2.imshow('img',img)
cv2.moveWindow('img',1075,800)
cv2.imshow('LAB',LAB)
cv2.moveWindow('LAB',1450,800)
#cv2.imshow('HSV',HLS)

cv2.waitKey(0)
cv2.destroyAllWindows()
