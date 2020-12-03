import cv2
import numpy as np

cap = cv2.VideoCapture(0)
def find_colors(image):
    global img
    size = 50
    areas = []
    num = 0
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    rgb = ["red","green","blue"]
    Color_range = [(0,0,255),(0,255,0),(255,0,0)]
    colors =  {(0,0,255):[(136, 87, 111),(180, 255, 255)],
          (0,255,0):[(65,60,60),(80,255,255)],
          (255,0,0):[(94, 80, 2),(120, 255, 255)]}
    for color,details in colors.items():
        hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv,details[0],details[1])
        object_edges =  cv2.Canny(mask,1500,150)
        contours,_ = cv2.findContours(object_edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        num += len(contours[:])
        areas.append(len(contours[:]))
    for i,area in enumerate(areas):
        percent = (area/num)*100
        cv2.putText(img,rgb[i]+f" {percent:.2f}",(20,size),cv2.FONT_HERSHEY_SIMPLEX,1,Color_range[i],4)
        size += 50
def filters(image):
    blur = cv2.GaussianBlur(image,(7,7),0)
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray,70,150)
def find_points(image,cnts):
    global img
    points = []
    xs,ys = [],[]
    for i in cnts[:]:
        xs.append(i[0][0][0])
        ys.append(i[0][0][1])
    if xs != [] or ys != []:
        points.append([min(xs),min(ys)])
        points.append([max(xs),max(ys)])
        if len(points) == 2:
            for i in points:
                cv2.circle(img,tuple(i),10,(255,255,255),-1)
            rect = clone[points[0][1]:points[1][1],points[0][0]:points[1][0]]
            try:
                find_colors(rect)
            except Exception:
                pass
        
while True:
    ret,img = cap.read()
    clone = img.copy()
    edges = filters(img)
    cnts,_ = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    find_points(img,cnts)
    cv2.imshow("img",img)
    key = cv2.waitKey(10)
    if key == ord("q"):
        break
cv2.destroyAllWindows()
cap.release()

