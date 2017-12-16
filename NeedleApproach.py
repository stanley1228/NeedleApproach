import numpy as np
import cv2

#架好攝影機，調焦
#把進給的地方包起來 or 利用去背景 將針點明顯化
#光流

#harris corner

filename = 'Needle2.jpg'
img = cv2.imread(filename)
print(img.shape)

cv2.namedWindow('Needle')
cv2.imshow("Needle",img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
#dst = cv2.cornerHarris(gray,2,3,0.05)
dst = cv2.cornerHarris(gray,blockSize=2,ksize=3,k=0.05)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
#the needle corner score is too low 
#original is 0.01*dst.max()  now change to 0.001*dst.max() then needle can be found
#i think it cause by the needle and the sewing machine gray part has similar color


img[dst>0.0001*dst.max()]=[0,0,255]  

cv2.imshow('harris',dst)
cv2.imshow('dst',img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
 


'''
filename = 'Needle2.jpg'
img = cv2.imread(filename)
print(img.shape)

cv2.imshow("Needle",img)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)

for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)

cv2.imshow("dst",img)

if cv2.waitKey(0) & 0xff == 27:
    cv2.destroyAllWindows()
'''

'''
#
#video
#
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    #height,width = frame.shape
    
    print(frame.shape)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

print('Hello World')
'''