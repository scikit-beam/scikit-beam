#Synthetic Shapes Module

#tri_2D = np.zeros((y_dim, x_dim))
##B = (200, 50)
#C = (300, 300)
import numpy as np
def draw_circle (src_img, 
                 center_coord=None, 
                 radius=None, 
                 value=None,
                 draw_exterior=None):
    y_dim, x_dim = src_img.shape
    mask_2D = np.zeros ((y_dim, x_dim))
    Y, X = np.ogrid[0:y_dim, 0:x_dim]
    if center_coord == None:
        center_y = (y_dim / 2)
        center_x = (x_dim / 2)
    else:
        center_y, center_x = center_coord
    if radius == None:
        radius = np.amin([y_dim, x_dim]) / 2
    circle_mask = (((X - center_x) ** 2) + (( Y - center_y) ** 2) < radius**2)
    exterior_mask = (((X - center_x) ** 2) + (( Y - center_y) ** 2) > radius**2)
    if draw_exterior == "YES":
        output_img = src_img * circle_mask
        if value != None:
            exterior_mask = exterior_mask * value
        output_img = output_img + exterior_mask
    else:
        output_img = src_img * exterior_mask
        if value != None:
            circle_mask = circle_mask * value
        output_img = output_img + circle_mask
    return output_img

def draw_elipse (src_img, 
                 y_distort, 
                 x_distort, 
                 center_y=None, 
                 center_x=None, 
                 obj_area=None):
    a = x_distort
    b = y_distort
    y_dim, x_dim = src_img.shape
    if center_y == None:
        center_y = (y_dim / 2)
    if center_x == None:
        center_x = (x_dim / 2)
    if obj_area == None:
        obj_area = (x_dim * y_dim / 4)
    mask_2D = np.zeros ((y_dim, x_dim))
    Y, X = np.ogrid[0:y_dim, 0:x_dim]
    mask_2D = ((X-center_x)**2/a**2)+((Y-center_y)**2/b**2) > obj_area
    return mask_2D

def draw_triangle_outline (src_img, 
                           A, 
                           B, 
                           C, 
                           value = 1):
    print "first coordinates: " + str(A)
    print "second coordinates: " + str(B)
    print "third coordinates: " + str(C)
    ya, xa = A
    yb, xb = B
    yc, xc = C
    x_coord_list = [xa, xb, xc]
    y_coord_list = [ya, yb, yc]
    for x_index in range(len(x_coord_list)):
        x1 = x_coord_list[x_index]
        y1 = y_coord_list[x_index]
        if x_index == len(x_coord_list)-1:
            x2 = x_coord_list[0]
            y2 = y_coord_list[0]
        else:
            x2 = x_coord_list[x_index + 1]
            y2 = y_coord_list[x_index + 1]
        print ("Drawing from: (" + str(y1) + ", " + str(x1) + ") to (" + 
               str(y2) + ", " + str(x2) + ")")
        x_list = [x1, x2]
        y_list = [y1, y2]
        x_range = np.arange(np.amin(x_list), np.amax(x_list))
        y_range = np.arange(np.amin(y_list), np.amax(y_list))
        x_len = len(x_range)
        y_len = len(y_range)
        x_coord = x1
        y_coord = y1
        if x_len >= y_len:
            step_sz = (float(y2-y1)/x_len)
            print "step size: " + str(step_sz)
            y_range = np.arange(y1,y2,step_sz)
            for i in y_range:
                y_coord = int(i)
                src_img[y_coord,x_coord] = value
                x_coord = x_coord + (x2-x1)/abs(x2-x1)
        elif y_len > x_len:
            step_sz = float(x2-x1)/y_len
            print "step size: " + str(step_sz)
            x_range = np.arange(x1,x2,step_sz)
            for i in x_range:
                x_coord = int(i)
                src_img[y_coord, x_coord] = value
                y_coord = y_coord + (y2-y1)/abs(y2-y1)
        src_img[y2, x2] = value
    return src_img

def draw_rectangle (src_img, vert_len, horiz_len, center, value=None):
    y_offset = vert_len/2
    x_offset = horiz_len/2
    Y, X = src_img.shape
    y_cen, x_cen = center
    if value == None:
        value = 1
    if ((y_cen - offset) < 0 or 
        (x_cen - offset) < 0 or 
        (y_cen + offset) > Y or 
        (x_cen + offset) > X):
        raise ValueError ("Object extends beyond image boundaries. Adjust " + 
                          "axial length and/or object center so that entire " +
                          "object fits within the source object boundaries. " + 
                          "Source object boundaries are " + 
                          "[VERTICAL, HORIZONTAL]: [0:" + 
                          str(np.amax(Y)) + ", 0:" + str(np.amax(X)) + "]")
    output_img = src_img[(y_cen-y_offset):(y_cen+y_offset),
                         (x_cen-x_offset):(x_cen+x_offset)] = value
    return output_img

def draw_square (src_img, axial_len, center, value=None):
    output_img = draw_rectangle(src_img, axial_len, axial_len, center, value)
    return output_img

def draw_sphere (src_img,
                  center_coord=None, 
                  radius=None, 
                  value=None,
                  draw_exterior=None):
    z_dim, y_dim, x_dim = src_img.shape
    mask_3D = np.zeros ((z_dim, y_dim, x_dim))
    Z, Y, X = np.ogrid[0:z_dim, 0:y_dim, 0:x_dim]
    if center_coord == None:
        center_z = (z_dim / 2)
        center_y = (y_dim / 2)
        center_x = (x_dim / 2)
    else:
        center_z, center_y, center_x = center_coord
    if radius == None:
        radius = np.amin([z_dim, y_dim, x_dim]) / 2
    sphere_mask = (((X - center_x) ** 2) + 
                   (( Y - center_y) ** 2) + 
                   (( Z - center_z) ** 2) 
                   < radius**2)
    exterior_mask = (((X - center_x) ** 2) + 
                     (( Y - center_y) ** 2) + 
                     (( Z - center_z) ** 2)
                     > radius**2)
    if draw_exterior == "YES":
        output_img = src_img * sphere_mask
        if value != None:
            exterior_mask = exterior_mask * value
        output_img = output_img + exterior_mask
    else:
        output_img = src_img * exterior_mask
        if value != None:
            sphere_mask = sphere_mask * value
        output_img = output_img + sphere_mask
    return output_img

def draw_cylinder (src_img, center_coord=None, radius=None, height=None, value=None, draw_exterior=None, apply_mask=None):
    if apply_mask == None or apply_mask == 'NO' or apply_mask == 'No' or apply_mask == 'no':
        mask_vol = np.zeros(src_img.shape)
    elif apply_mask == 'YES' or apply_mask == 'Yes' or apply_mask == 'yes':
        mask_vol = src_img
    else:
        raise ValueError('Verify setting for apply_mask kwarg.')
    if height == None:
        cylinder_h = mask_vol.shape[0]
        for x in range(cylinder_h):
            #slc = np.zeros((mask_vol.shape[1], mask_vol.shape[2]))
            slc = mask_vol[x,:,:]
            mask_vol[x,:,:] = draw_circle(slc, center_coord, radius, value, draw_exterior)
    return mask_vol