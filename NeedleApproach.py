import numpy as np
import cv2

cap = cv2.VideoCapture(0)

ret,img1 =cap.read()

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray1 = np.float32(gray1)
Hcorner1 = cv2.cornerHarris(gray1,blockSize=2,ksize=3,k=0.05)
 
#dilated for marking the corners, not important
feature_params = dict( maxCorners = 20,
                    qualityLevel = 0.2,
                    minDistance = 40,
                    blockSize = 2,
                    useHarrisDetector=False,
                    k=0.05)

gray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
GFeature1 = cv2.goodFeaturesToTrack(gray1,**feature_params)
goodFeature_result=img1.copy()

for i in GFeature1:
    x,y = i.ravel()
    cv2.circle(goodFeature_result,(x,y),3,255,-1)#plot circle

#=================
#find needle point
#================= 
NeedleGet=False
while(True):   
    ret,img2=cap.read()
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    FlowPoint1, status, err=cv2.calcOpticalFlowPyrLK(gray1,gray2,GFeature1,None,**lk_params)

    #Find a point that move the most far from last frame and current frame
    #the point is needle point
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
     
        x,y=FlowPoint1[max_index].ravel()
        NeedlePoint=(x,y)
        NeedleGet=True
        ss="Needle={:.3f},{:.3f}".format(NeedlePoint[0],NeedlePoint[1])
     
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

    #record the needle coordinate
    if NeedleGet==True:
        NeedlePoint=(x,y)

    #wait space key to change to find the symbol 
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

#=========================
#find the symbol on object
#=========================
SymbolGet=False
while(True):
    ret,img2=cap.read()
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    FlowPoint2, status, err=cv2.calcOpticalFlowPyrLK(gray1,gray2,GFeature2,None,**lk_params)

    #Find a point that move the most far from last frame and current frame
    #the point is needle point
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

    #check the offset between needle point and object
    #and give the direction hint
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
        
    #show point pixel on the image
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