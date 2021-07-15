import  cv2
import numpy as np
import math
import datetime
capture=cv2.VideoCapture(0)
while capture.isOpened():
    _,frame=capture.read()

    cv2.rectangle(frame,(100,100),(300,300),(0,255,0),0)
    crop_image=frame[100:300,100:300]

    blur=cv2.GaussianBlur(crop_image,(3,3),0)

    hsv=cv2.cvtColor(blur,cv2.COLOR_BGR2HSV)
    mask2=cv2.inRange(hsv,np.array([2,0,0]),np.array([20,255,255]))

    karnel=np.ones((5,5))
    dilation=cv2.dilate(mask2,karnel,iterations=1)
    erosion=cv2.erode(dilation,karnel,iterations=1)

    filtered=cv2.GaussianBlur(erosion,(3,3),0)

    _,tresh=cv2.threshold(filtered,150,255,cv2.THRESH_BINARY)
    cv2.imshow("Theashold",tresh)

    conturs,_=cv2.findContours(tresh.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    try:
        contur=max(conturs,key= lambda x:cv2.contourArea(x))

        x,y,w,h=cv2.boundingRect(contur)
        cv2.rectangle(crop_image,(x,y),(x+w,y+h),(0,255,255),0)

        hull=cv2.convexHull(contur)

        drawing=np.zeros((200,200,3), np.uint8)
        cv2.drawContours(drawing,[contur],-1,(0,255,0),0)
        cv2.drawContours(drawing,[hull],-1,(255,0,255),0)

        hull=cv2.convexHull(contur,returnPoints=False)
        defect=cv2.convexityDefects(contur,hull)

        count_defects=0
        # print(defect.shape)
        for i in range(defect.shape[0]):


            s,e,f,d=defect[i,0]
            start=tuple(contur[s,0])
            end=tuple(contur[e,0])
            far=tuple(contur[f,0])
            # print(f"start={start}")
            # print(f"end={end}")
            # print(f"far={far}")
            a=math.sqrt((end[0]-start[0])**2+(end[1]-start[1])**2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle=(math.acos((b**2+c**2-a**2)/(2*b*c))*180)/3.14

            if angle<=90:
                count_defects+=1
                cv2.circle(crop_image,far,1,[0,255,0],-1)

            cv2.line(crop_image,start,end,[0,255,0],2)

        if count_defects==0:
            cv2.putText(frame,"ONE",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        elif count_defects == 1:
            cv2.putText(frame, "TWO", (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 2)
        elif count_defects==2:
            cv2.putText(frame,"THREE",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        elif count_defects==3:
            cv2.putText(frame,"FOUR",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        elif count_defects==4:
            cv2.putText(frame,"FIVE",(50,50),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        else:
            pass

    except:
        pass

    cv2.imshow("GESTURE",frame)
    # all_image=np.hstack(drawing,crop_image)
    cv2.imshow("CONTURE",crop_image)
    cv2.imshow("Drawing",drawing)
    if cv2.waitKey(1)&0xFF==ord("q"):
        break
capture.release()
cv2.destroyAllWindows()
