from __future__ import print_function
import cv2
import numpy as np
import csv

drawing=False # true if mouse is pressed
# mode=True # if True, draw rectangle. Press 'm' to toggle to curve

INITIAL = []
TRAJ=[]

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing, mode
    global img
    global INITIAL,TRAJ

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
        print ("drawing =", drawing)
        if len(INITIAL) == 0:
            INITIAL = [ix, iy]
            print (INITIAL)

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(img, (ix, iy), (x, y), (0, 0, 255), 10)
            ix = x
            iy = y
            TRAJ.append([ix,iy])


    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False

def write_file(data_list, file_name):
    # emg: data[1] ; imu: data[2]; cls: data[3]
    print ("write to csv")
    with open(file_name,'a') as csvfile:
        data_writer = csv.writer(csvfile, delimiter=' ',quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for data in data_list:
            data_writer.writerow(data)


def main():
    global img
    img = np.zeros((512,512,3), np.uint8)
    save_index = 0
    global INITIAL
    global TRAJ
    file_name = './two_dim/trajectory'

    cv2.namedWindow('paint_board')
    cv2.setMouseCallback('paint_board',interactive_drawing)
    while(1):
        cv2.imshow('paint_board',img)
        k=cv2.waitKey(1)&0xFF
        if k==ord('q'):
            break
        elif k == ord('s'):
            save_index += 1
            print ("save current image, index is", save_index)
            print (INITIAL)
            print (TRAJ)

    #         todo: save all the points
        elif k ==ord('c'):
            print ("clear the image")
            img = np.zeros((512, 512, 3), np.uint8)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
