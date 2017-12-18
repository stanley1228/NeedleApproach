import numpy as np
import cv2

#?[?n???v????A???J
#?????i?????a???]?X_?? or ???Q???h?I?? ??N?Xw?I??????
#???y


'''
add optical flow
'''
cap = cv2.VideoCapture(0)

ret,img1 =cap.read()

cv2.imshow("img1",img1)

gray = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
Hcorner1 = cv2.cornerHarris(gray,blockSize=2,ksize=3,k=0.05)

harris_result=img1.Copy()
harris_result[Hcorner1>0.2*Hcorner1.max()]=[0,0,255]  
cv2.imshow('harris',harris_result)
cv2.waitKey(1)

#find needle point 
while(True):   
    ret,img2=cap.read()
    # Parameters for lucas kanade optical flow
    #lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    Hcorner2, status, err=cv2.calcOpticalFlowPyrLK(img1,img2,Hcorner1,None)

    # Select good points ==> status is 1 and moving a enough distnace  
    k=0
    for i in len(Hcorner2):
        if status[i]==1 and (abs(Hcorner2[i].x-Hcorner1[i].x)+abs(Hcorner2[i].y-Hcorner1[i].y)>4):
            Hcorner2[k]=Hcorner2[i]
            k=k+1
    
    #record the needle coordinate and break this loop
    if k==1 or (cv2.waitKey(100) & 0xFF == ord('q')):
        NeedlePoint = HCorner2[0]
        break

    img1=img2


 #find the symbol on object
while(True):
    ret,img2=cap.read()

    # Parameters for lucas kanade optical flow
    #lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    Hcorner2, status, err=cv2.calcOpticalFlowPyrLK(img1,img2,Hcorner1,None)

    # Select good points ==> status is 1 and moving a enough distnace  
    k=0
    for i in len(Hcorner2):
        if status[i]==1 and (abs(Hcorner2[i].x-Hcorner1[i].x)+abs(Hcorner2[i].y-Hcorner1[i].y)>4):
            Hcorner2[k]=Hcorner2[i]
            k=k+1
    
    #find the symbol coordinate 
    if k==1:
        SymbolPoint = HCorner2[0]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    offset=SymbolPoint-NeedlePoint

    if(offset.x)>0:
        print('move left')
    else:
        print('move right')
    
    if(offset.y)>0:
        print('move down')
    else:
        print('move up')

    img1=img2



    

    

        
     

'''
#
#harris corner
#
cap = cv2.VideoCapture(0)
while(True):
    ret, img = cap.read()

    print(img.shape)
    #filename = 'Needle2.jpg'
    #img = cv2.imread(filename)
    #print(img.shape)

    cv2.namedWindow('Needle')
    cv2.imshow("Needle",img)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,blockSize=2,ksize=3,k=0.05)
   
    #result is dilated for marking the corners, not important
    #dst = cv2.dilate(dst,None)
    kernel = np.ones((8,8), np.uint8)
    dst = cv2.dilate(dst,kernel,iterations=1)

    # Threshold for an optimal value, it may vary depending on the image.
    #the needle corner score is too low 
    #original is 0.01*dst.max()  now change to 0.001*dst.max() then needle can be found
    #i think it cause by the needle and the sewing machine gray part has similar color
    img[dst>0.2*dst.max()]=[0,0,255]  
    cv2.imshow('harris',dst)
    cv2.imshow('dst',img)

    
    cv2.calcOpticalFlowPyrLK(I1,I2,features,features_after,status,err);
    








    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
'''

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


#
#video
#
'''
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