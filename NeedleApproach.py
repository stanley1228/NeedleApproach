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

#cv2.imshow("img1",img1)
#cv2.waitKey(1)

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
Hcorner1 = cv2.cornerHarris(gray1,blockSize=2,ksize=3,k=0.05)
 
#dilated for marking the corners, not important
kernel = np.ones((8,8), np.uint8) 
Hcorner1 = cv2.dilate(Hcorner1,kernel,iterations=1)

#harris_result=img1.copy()
#harris_result[Hcorner1>0.2*Hcorner1.max()]=[0,0,255]  
#cv2.imshow('harris',harris_result)
#cv2.waitKey(1)

feature_params = dict( maxCorners = 20,
                    qualityLevel = 0.2,
                    minDistance = 40,
                    blockSize = 2,
                    useHarrisDetector=False,
                    k=0.05)

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
#GFeature1 = cv2.goodFeaturesToTrack(gray1,25,0.01,10)
GFeature1 = cv2.goodFeaturesToTrack(gray1,**feature_params)

goodFeature_result=img1.copy()

for i in GFeature1:
    x,y = i.ravel()
    cv2.circle(goodFeature_result,(x,y),3,255,-1)

#cv2.imshow('first_get',goodFeature_result)
#cv2.waitKey(1)


#find needle point 
NeedleGet=False
while(True):   
    ret,img2=cap.read()
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    FlowPoint1, status, err=cv2.calcOpticalFlowPyrLK(gray1,gray2,GFeature1,None,**lk_params)

    # Select good points ==> status is 1 and moving a enough distnace  
    runtime=len(FlowPoint1)
    dist=[]
    for i in range(runtime):
        x,y=FlowPoint1[i].ravel()
        x2,y2=GFeature1[i].ravel()
        dist.append(abs(x2-x)+abs(y2-y))
        
    max_dist=max(dist)
    max_index=dist.index(max_dist)
    if status[max_index]==1 and (max_dist>20):
        ss="distance={:.3f}".format(max_dist)
        #print(ss)
     
        x,y=FlowPoint1[max_index].ravel()
        NeedlePoint=(x,y)
        NeedleGet=True
        ss="Needle={:.3f},{:.3f}".format(NeedlePoint[0],NeedlePoint[1])
        #print(ss)
        if len(FlowPoint1)>1:
            GFeature1=FlowPoint1[max_index]
        else:
            GFeature1=FlowPoint1.copy()
        gray1=gray2.copy()

    #show tracking feature
    goodFeature_result=img2.copy()
    if NeedleGet==True:
        sNeedlePoint="{:.1f},{:.1f}".format(NeedlePoint[0],NeedlePoint[1])
        cv2.putText(goodFeature_result,sNeedlePoint, org=NeedlePoint,fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,0,255),lineType=2)

    for i in FlowPoint1:
        x,y = i.ravel()
        cv2.circle(goodFeature_result,(x,y),3,255,-1)
    cv2.imshow('goodFeature1',goodFeature_result)    
    cv2.waitKey(100)

    if NeedleGet==True:
        NeedlePoint=(x,y)

    #record the needle coordinate and break this loop
    if (cv2.waitKey(100) & 0xFF == ord(' ')):
        print('start symbol detect')
        cv2.destroyWindow('goodFeature1')
        cv2.waitKey(100)
        break

 
    
ret,img1 =cap.read()
feature_params = dict( maxCorners = 20,qualityLevel = 0.1, minDistance = 50,blockSize = 2,useHarrisDetector=False,k=0.05)
gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
GFeature2 = cv2.goodFeaturesToTrack(gray1,**feature_params)

for i in GFeature2:
    x,y = i.ravel()
    cv2.circle(goodFeature_result,(x,y),3,255,-1)

cv2.imshow('first_get',goodFeature_result)
cv2.waitKey(1)

        
#find the symbol on object
SymbolGet=False
while(True):
    ret,img2=cap.read()
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    FlowPoint2, status, err=cv2.calcOpticalFlowPyrLK(gray1,gray2,GFeature2,None,**lk_params)

    # Select good points ==> status is 1 and moving a enough distnace  
    runtime=len(FlowPoint2)
    dist=[]
    for i in range(runtime):
        x,y=FlowPoint2[i].ravel()
        x2,y2=GFeature2[i].ravel()
        dist.append(abs(x2-x)+abs(y2-y))
        
    max_dist=max(dist)
    max_index=dist.index(max_dist)
    if status[max_index]==1 and (max_dist>20):
        x,y=FlowPoint2[max_index].ravel()
        SymbolPoint=(x,y)
        SymbolGet=True
       
        if len(FlowPoint2)>1:
            GFeature2=FlowPoint2[max_index]
        else:
            GFeature2=FlowPoint2.copy()
        gray1=gray2.copy()


    if SymbolGet==True:
        SymbolPoint=(x,y)
 
        offset_x=SymbolPoint[0]-NeedlePoint[0]
        offset_y=SymbolPoint[1]-NeedlePoint[1]

        if(offset_x)>0:
            ss='move left'
        else:
            ss='move right'
    
        if(offset_y)>0:
            ss=ss+' up'
        else:
            ss=ss+' down'

        if(offset_x<20) and (offset_y<20):
            ss='ok!'
        
    #show tracking feature
    match_symbol=img2.copy()
    cv2.circle(match_symbol,NeedlePoint,3,(0,0,255),-1)
    if SymbolGet==True:
        sSymbolPoint="{:.1f},{:.1f}".format(SymbolPoint[0],SymbolPoint[1])
        sNeedlePoint="{:.1f},{:.1f}".format(NeedlePoint[0],NeedlePoint[1])
        cv2.putText(match_symbol,sSymbolPoint, org=SymbolPoint,fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,0,255),lineType=2)
        cv2.putText(match_symbol,sNeedlePoint, org=NeedlePoint,fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,color=(0,0,255),lineType=2)
        cv2.putText(match_symbol,ss, org=(10,50),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(255,255,0),lineType=2)
    
    for i in FlowPoint2:
        x,y = i.ravel()
        cv2.circle(match_symbol,(x,y),3,255,-1)
    cv2.imshow('match_symbol',match_symbol)    
    cv2.waitKey(100)

    if SymbolGet==True:
        SymbolPoint=(x,y)


    

    
 
#
#harris corner
#
'''
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
    img[dst>0.1*dst.max()]=[0,0,255]  
    cv2.imshow('harris',dst)
    cv2.imshow('dst',img)

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