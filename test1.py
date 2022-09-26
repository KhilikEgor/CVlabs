# ITMO University
# Mobile Computer Vision course
# 2020
# by Aleksei Denisov
# denisov@itmo.ru

import cv2
import numpy as np

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


#def rectangle_control():


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def colors_masks(hsvFrame):
    # define mask
    red_lower = np.array([136, 87, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    green_lower = np.array([75, 50, 80], np.uint8)
    green_upper = np.array([90, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
    return red_mask,green_mask,blue_mask

def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    print(gstreamer_pipeline(flip_method=4))
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    #cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)
    start_point = (0, 0)
    end_point = (100, 100)
    color = (0, 255, 0)
    thickness = 1
    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, frame = cap.read()

            invert = ~frame

            # Show video
            #cv2.imshow('Inverted', invert)

            cropped_frame = frame[start_point[1]:end_point[1], start_point[0]:end_point[0]]
            hsvFrame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2HSV)
            masks = colors_masks(hsvFrame)
            cv2.rectangle(frame, start_point, end_point, color, thickness)
            kernal = np.ones((5,5), "uint8")

            red_mask = cv2.dilate(masks[0], kernal)
            cv2.bitwise_and(cropped_frame, cropped_frame, mask = masks[0])
            red_contours, hierarchy = cv2.findContours(red_mask.copy(),
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
           # cv2.drawContours(cropped_frame, red_contours, -1, (0, 0, 255), 3)
            for pic, contour in enumerate(red_contours):
                area = cv2.contourArea(contour)
                print(area)
                if (area > 4000):
                    cv2.putText(frame, "RED", (start_point[0] + 10,end_point[1] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 0, 255),thickness = 2)

            blue_mask = cv2.dilate(masks[2], kernal)
            cv2.bitwise_and(cropped_frame, cropped_frame, mask = blue_mask)
            blue_contours, hierarchy = cv2.findContours(blue_mask.copy(),
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(cropped_frame, blue_contours, -1, (255, 0, 0), 3)
            for pic, contour in enumerate(blue_contours):
                area = cv2.contourArea(contour)
                print(area)
                if (area > 4000):
                    cv2.putText(frame, "BLUE", (start_point[0] + 10,end_point[1] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 0, 0),thickness = 2)


            green_mask = cv2.dilate(masks[1], kernal)
            cv2.bitwise_and(cropped_frame, cropped_frame, mask=green_mask)
            green_contours, hierarchy = cv2.findContours(green_mask.copy(),
                                            cv2.RETR_TREE,
                                            cv2.CHAIN_APPROX_SIMPLE)
            #cv2.drawContours(cropped_frame, green_contours, -1, (0, 255, 0), 3)
            for pic, contour in enumerate(green_contours):
                area = cv2.contourArea(contour)
                print(area)
                if (area > 4000):
                    cv2.putText(frame, "GREEN", (start_point[0] + 10,end_point[1] + 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (0, 255, 0), thickness = 2)


            cv2.imshow('Inverted', frame)

            # This also acts as
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break

            # Left arrow
            if keyCode == 81 and start_point[0] >= 0 and end_point[0] >= 0:
                start_point = (start_point[0] - 15, start_point[1])
                end_point = (end_point[0] - 15, end_point[1])
            # Up arrow
            if keyCode == 82 and start_point[1] >= 0 and end_point[1] >= 0:
                start_point = (start_point[0], start_point[1] - 15)
                end_point = (end_point[0], end_point[1] - 15)

            # Right arrow
            if keyCode == 83 and start_point[0] <= 1280 and end_point[0] <= 1280:
                start_point = (start_point[0] + 15, start_point[1])
                end_point = (end_point[0] + 15, end_point[1])

            # Down arrow
            if keyCode == 84 and start_point[1] <= 720 and end_point[1] <= 720:
                start_point = (start_point[0], start_point[1] + 15)
                end_point = (end_point[0], end_point[1] + 15)

        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()