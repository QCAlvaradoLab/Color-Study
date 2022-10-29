import re
import cv2
import os 
import numpy as np

class HSVColorClassifier(object):
        
    mouse_x, mouse_y = None, None
    current_range = [[180,255,255], [0,0,0]]

    def __init__(self, color_palette_path, labels_ranges_path="./resources/colors.txt", gui=True):
        
        img = cv2.imread(color_palette_path)
        
        with open(labels_ranges_path, 'r') as f:
            self.colors = [x.strip() for x in f.readlines()]
            self.color_intervals = [re.findall(r"\(([a-z_]+)\,\s(.+)\)", x) for x in self.colors]
            self.color_intervals = [x[0] for x in self.color_intervals if len(x) > 0]
            
            self.color_intervals = [(x[0], eval(x[1])) for x in self.color_intervals]

        self.hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print (self.hsv_img.shape)

        if gui:

            cv2.imshow('image', img)
            cv2.namedWindow('image')
            cv2.setMouseCallback('image', self.get_mouse_coords)

            cv2.waitKey(0)

    def get_mouse_coords(self, event, x, y, flags, param):
        
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_x = x
            self.mouse_y = y
        
            color_pt = self.hsv_img[y, x, :]
            
            for idx in range(3):
                if color_pt[idx] < self.current_range[0][idx]:
                    self.current_range[0][idx] = color_pt[idx]
                if color_pt[idx] > self.current_range[1][idx]:
                    self.current_range[1][idx] = color_pt[idx]
            
            print (self.get_color(color_pt))

        elif event == cv2.EVENT_LBUTTONDBLCLK:
            print (self.current_range)
        
        if cv2.getWindowProperty('image', 0) < 0:
            exit()

    def get_color(self, hsv_coord):
        ''' 
        if hsv_coord.shape == 1:
            hsv_coord = np.array([[hsv_coord]]).astype(np.uint8)
        else:
            hsv_coord = np.array(hsv_coord).astype(np.uint8)      
        '''
        
        if len(hsv_coord.shape) == 1:
            hsv_coord = np.array([[hsv_coord]]).astype(np.uint8)
            vectorized = False
        else:
            hsv_coord = np.array(hsv_coord).astype(np.uint8)
            vectorized = True
            mask = np.zeros_like(hsv_coord[:,0])
        
        grid_w = np.sqrt(hsv_coord.shape[0])
        grid_h = round(grid_w) + 1
        grid_w = round(grid_w) + 1
        
        color_grid = np.zeros((grid_h, grid_w, 3))
        pad = grid_h * grid_w - hsv_coord.shape[0]

        print ('1', pad, grid_h*grid_w - pad, hsv_coord.shape[0])
        hsv_coord = np.pad(hsv_coord, ((pad//2, 0), (0, 0)))
        print ('2', pad, grid_h*grid_w - pad, hsv_coord.shape[0])
        print (hsv_coord.shape[0], ) 

        print ('3', hsv_coord.shape, np.prod(hsv_coord.shape) // 3)
        print ('4', grid_h, grid_w, np.prod(hsv_coord.shape), grid_h - pad//2, grid_w - pad//2)
        color_grid[:grid_h, :grid_w, :] = hsv_coord.reshape((grid_h, grid_w, 3))

        bool_fn = lambda begin_range, end_range: cv2.inRange(
                    np.array(hsv_coord).astype(np.uint8), np.array(begin_range), np.array(end_range))
       
        for idx, color_pair in enumerate(self.color_intervals):
            
            color_bool_map = bool_fn(*color_pair[1])

            if not vectorized and all(color_bool_map):
                return color_pair[0]
            
            if vectorized:
                print (idx, color_bool_map.max())
                mask += ((idx+1) * color_bool_map.astype(np.uint8)) // 255

                cv2.imshow("mask", mask*255, cmap="red")
                cv2.waitKey()
                
                colors, counts = np.unique(mask, return_counts=True) 
                print (colors, counts)
                
        if not vectorized:
            return "background"
        else:
            return mask

if __name__ == "__main__":
    
    '''
    # INSTRUCTIONS:

    Left Click - Add to color range per window
    Right Click - Display current cumulative color range
    '''

    imgpath = "./resources/palette.png"
    
    color_classifier = HSVColorClassifier(imgpath, gui=False)
    print (color_classifier.get_color(color_classifier.hsv_img)) 
