# opencv-advanced-color-detection
This program is supposed to find an object in a frame and show the percentange of rgb(red-green-blue) colors used in the specific object.
![](https://github.com/Moeed1mdnzh/opencv-advanced-color-detection/blob/main/example1.jpg)
![](https://github.com/Moeed1mdnzh/opencv-advanced-color-detection/blob/main/example2.jpg)
![](https://github.com/Moeed1mdnzh/opencv-advanced-color-detection/blob/main/example3.jpg)
# STEPS
Part_1
1_Captures the frame,clones it and adds filters such as blurring and converting to grayscale to it.
2_Finds the edges and contours.
Part_2
1_Finds the smallest contour and the biggest contour and draws a circle on those points.
2_Crops it.
Part_3
1_Detects colors by the order of rgb(red-green-blue) colors and creates a mask.
2_Finds the edges of the mask and then finds the contours and gets the length of them.
3_Divides the length of each color by the total of color lengths and then multiplies it by 100 so we can get the percentage of each color.
4_Displays the frame
# Requirements
opencv-python , numpy
