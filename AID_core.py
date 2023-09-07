# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:56:21 2019

@author: Borg
"""

import numpy as np
from PIL import Image, ImageDraw, ImageOps

import os
#import cv2 as cv
from scipy import ndimage
import json

MAGNIFICATION = ['4x',    '10x',   '20x',    '40x',    '60x']
PIXEL_AREA =    [2.49219, 0.39875, 0.099688, 0.024922, 0.011076] # um^2


class AID(object):
    def __init__(self, array_1, array_2, trashold, max_val, n_of_colors, mag, rad_to_pg):
        self.array_1 = array_1
        self.array_2 = array_2
        self.trashold = trashold
        self.max_val = max_val
        self.n_of_colors = n_of_colors
        self.mag = mag
        self.rad_to_pg = rad_to_pg
        
        self.EPS = 0.001
        
        self.red = np.zeros_like(array_1)
        self.green = np.zeros_like(array_1)
        self.blue = np.zeros_like(array_1)
        self.AID_raw = np.fliplr(np.rot90(np.zeros_like((array_1, array_1, array_1)), -1))
        self.AID_image_8b = object
        
        self.zero_line = np.zeros_like(array_1)
        self.zero_line_AID_8b = object
        self.zero_line_AID_raw = np.fliplr(np.rot90(np.zeros_like((array_1, array_1, array_1)), -1))
        self.zero_line_len = 0
        
        self.decrement_area = np.zeros(self.array_1.shape).astype(np.uint8)
        self.increment_area = np.zeros(self.array_1.shape).astype(np.uint8)
        self.anchor_area = np.zeros(self.array_1.shape).astype(np.uint8)
        self.anchor_decrement_area = np.zeros(self.array_1.shape).astype(np.uint8)
        self.anchor_increment_area = np.zeros(self.array_1.shape).astype(np.uint8)
        self.zero_line_area = np.zeros_like(array_1)
        
        self.areas_masks_8b = object
        self.areas_masks = np.fliplr(np.rot90(np.zeros_like((array_1, array_1, array_1)), -1))
        self.areas_masks_extended_8b = object
        self.areas_masks_extended = np.fliplr(np.rot90(np.zeros_like((array_1, array_1, array_1)), -1))
        self.extended_areas_masks = np.fliplr(np.rot90(np.zeros_like((array_1, array_1, array_1)), -1))
        self.extended_areas_masks_8b = object
        
        if self.rad_to_pg == True:
            if self.mag == MAGNIFICATION[0]:
                self.pixelArea = PIXEL_AREA[0]
            if self.mag == MAGNIFICATION[1]:
                self.pixelArea = PIXEL_AREA[1]
            if self.mag == MAGNIFICATION[2]:
                self.pixelArea = PIXEL_AREA[2]
            if self.mag == MAGNIFICATION[3]:
                self.pixelArea = PIXEL_AREA[3]
            if self.mag == MAGNIFICATION[4]:
                self.pixelArea = PIXEL_AREA[4]
        else:
            self.pixelArea = 1
        
        self.data = dict
        
        self.AID_image_8b_COG = object
        self.compose_image_8b_COG = object

        
    def diff(self, array_1, array_2):
        """ Returns difference between two arrays.
        """
        result = array_2 - array_1
        return result
    
    def compare_for_blue(self):
        """ Select new areas with non-zero values in the red and green channel.
        """
        result_r = (self.red - self.array_2 > self.EPS) * self.red # discard negative values from difference of red channel and array 2
        result_g = (self.green - self.array_1 > self.EPS) * self.green # discard negative values from difference of green channel and array 1
        return result_g+result_r
    
    def calculate_R_channel(self):
        """ Calculates red channel from two arrays.
        """
        self.red = self.diff(self.array_2, self.array_1) # red channel
        self.red = (self.red > self.trashold) * self.red # discard negative values from red channel

    def calculate_G_channel(self):
        """ Calculates green channel from two arrays.
        """
        self.green = self.diff(self.array_1,self.array_2) # green channel
        self.green = (self.green > self.trashold) * self.green # discard negative values from green channel

    def calculate_B_channel(self):
        """ Calculates blue channel from two arrays.
        """
        self.blue = self.compare_for_blue()
        
    def calculate_layer(self, array):
        """ Calculates the AID layer for the 8-bit image.
        """
        np.seterr(divide='ignore', invalid='ignore')
        max_array = array.max()
        min_array = array.min()
        diff_array = max_array-min_array
        return ((array/(diff_array*self.max_val))*255).astype(np.uint8)        
        
    def make_AID_image(self):
        """ Calculates the AID as an 8-bit image.
        """
        red = self.calculate_layer(self.red)
        green = self.calculate_layer(self.green)
        blue = self.calculate_layer(self.blue)

        if self.n_of_colors == 'two':
            AID = np.array([red, green, np.zeros_like(red)]).T
        if self.n_of_colors == 'four':
            AID = np.array([red, green, blue]).T
            
        self.AID_image_8b = Image.fromarray(AID, 'RGB')
        
    def make_AID_RAW(self):
        """ Calculates the AID as a raw quantitative 3D array.
        """
        AID_raw = np.array([self.red, self.green, self.blue]).T
        self.AID_raw = np.fliplr(np.rot90(AID_raw,-1))

    def get_AID_image(self):
        """ Returns the AID as an 8-bit image and a raw quantitative 3D array.
        """        
        self.calculate_R_channel()  
        self.calculate_G_channel()  
        self.calculate_B_channel()  
        self.make_AID_image()
        self.make_AID_RAW()
        return self.AID_image_8b, self.AID_raw
    
    def make_zero_line(self):
        """ Calculates the zero-line between the red and green parts of AID.
        """
        zero_line_len = 0
        for index_x in range(1, self.array_1.shape[0]-1):
            for index_y in range(1, self.array_1.shape[1]-1):
                value_red = self.red[index_x-1:index_x+1, index_y-1:index_y+1].sum()
                value_green = self.green[index_x-1:index_x+1, index_y-1:index_y+1].sum()
                if value_red > self.EPS and value_green > self.EPS:
                    self.zero_line[index_x, index_y] = 255
                    zero_line_len = zero_line_len + 1
                else:
                    pass
        self.zero_line_len = zero_line_len * np.sqrt(self.pixelArea)
                
    def make_zero_line_AID_8b(self):
        """ Calculates the AID with zero-line in the blue channel, 8-bit image.
        """
        red = self.calculate_layer(self.red)
        green = self.calculate_layer(self.green)
        blue = (self.zero_line).astype(np.uint8)
        zero_line = blue.T
        AID = np.array([red, green, np.zeros_like(red)]).T
        for z in range(AID.shape[2]):
            for i in range(AID.shape[0]):
                for j in range(AID.shape[1]):
                    if zero_line[i, j] == 255:
                        AID[i, j] = 255
                    else:
                        pass
        self.zero_line_AID_8b = Image.fromarray(AID, 'RGB')
        
    def make_zero_line_AID_raw(self):
        """ Calculates the AID with zero-line in the blue channel, raw quantitative 3D array.
        """
        max_red = np.amax(self.red)
        max_green = np.amax(self.green)
        max_value = max([max_red, max_green])
        blue = (self.zero_line).astype(np.float32) * max_value /255
        
        zero_line_AID = np.array([self.red, self.green, blue]).T
        self.zero_line_AID_raw = np.fliplr(np.rot90(zero_line_AID,-1))
    
    
    def get_zero_line_AID(self):
        """ Returns the AID (with zero-line in the blue channel) as an 8-bit image and a raw quantitative 3D array.
        """
        self.make_zero_line()
        self.make_zero_line_AID_8b()
        self.make_zero_line_AID_raw()
        return self.zero_line_AID_8b, self.zero_line_AID_raw

        
    def make_areas(self):
        """ Selects the anchor area, areas with decrement and increment.
        """
        for index_x in range(0, self.array_1.shape[0]):
            for index_y in range(0, self.array_1.shape[1]):
                if self.array_1[index_x, index_y] > self.EPS and self.array_2[index_x, index_y] == 0:
                    self.decrement_area[index_x, index_y] = 255
                if self.array_1[index_x, index_y] == 0 and self.array_2[index_x, index_y] > self.EPS:
                    self.increment_area[index_x, index_y] = 255
                if self.array_1[index_x, index_y] > self.EPS and self.array_2[index_x, index_y] > self.EPS:
                    self.anchor_area[index_x, index_y] = 255
                else:
                    pass
                
    def make_areas_with_extended_anchor(self):
        """ Selects the decrement and increment areas in anchor area. Selects zero-line in anchor area.
        """
        self.make_areas()
        
        for index_x in range(1, self.array_1.shape[0]-1):
            for index_y in range(1, self.array_1.shape[1]-1):
                value_red = self.red[index_x-1:index_x+1, index_y-1:index_y+1].sum()
                value_green = self.green[index_x-1:index_x+1, index_y-1:index_y+1].sum()
                if value_red > self.EPS and value_green > self.EPS and self.anchor_area[index_x, index_y] == 255:
                    self.zero_line_area[index_x, index_y] = 255
                if value_red > self.EPS and value_green < self.EPS and self.anchor_area[index_x, index_y] == 255:
                    self.anchor_decrement_area[index_x, index_y] = 255
                if value_red < self.EPS and value_green > self.EPS and self.anchor_area[index_x, index_y] == 255:
                    self.anchor_increment_area[index_x, index_y] = 255
                else:
                    pass
        
    def get_extended_anchor_areas(self):
        """ Returns the extended masks with zero-line as an 8-bit image and a raw quantitative 3D array.
        """
        self.make_areas_with_extended_anchor()
        base_areas_masks = np.array([self.decrement_area, self.increment_area, self.anchor_area]).T
        anchor_areas_masks = np.array([self.anchor_decrement_area, self.anchor_increment_area, np.zeros_like(self.array_1)]).T
        zero_line = np.array([self.zero_line_area, self.zero_line_area, self.zero_line_area]).T
        masks = np.add(base_areas_masks, anchor_areas_masks)
        masks = np.add(masks, zero_line)
        self.extended_areas_masks = (masks).astype(np.uint8)
        self.extended_areas_masks_8b = Image.fromarray(self.extended_areas_masks, 'RGB')
        return self.extended_areas_masks_8b, self.extended_areas_masks
    
    def get_area_masks(self):
        """ Returns the basic masks as an 8-bit image and a raw quantitative 3D array.
        """
        self.make_areas()
        self.areas_masks = np.array([self.decrement_area, self.increment_area, self.anchor_area]).T
        self.areas_masks_8b = Image.fromarray(self.areas_masks, 'RGB')
        return self.areas_masks_8b, self.areas_masks  

    def calculate_cog(self, array, weight=False):
        """ Calculates the centre of gravity of the numpy array. Weighting is possible.
        """
        pixel_size = np.sqrt(self.pixelArea)
                
        if weight == True:
            cog = tuple([pixel_size*i for i in ndimage.measurements.center_of_mass(array)])
        if weight == False:
            array = (array > self.EPS) * 1
            cog = tuple([pixel_size*i for i in ndimage.measurements.center_of_mass(array)])
        else:
            pass       
        return cog

    def calculate_area_size(self, array, weight=False):
        """ Calculates the size of non-zero area in the numpy array.
        If weighted, returns area in um^2, else its return a count of nonzero values.
        """
        pixel_size = np.sqrt(self.pixelArea)
        size = 0
        for index_x in range(0, self.array_1.shape[0]):
            for index_y in range(0, self.array_1.shape[1]):
                if array[index_x, index_y] > self.EPS:
                    size = size + 1
                else:
                    pass

        if weight == True:
            size = size * pixel_size
        else:
            pass
        return size

    def make_masked_part_of_AID(self):
        """ Makes data dictionary of areas, included mass, centres of gravities.
        """
        #AID_projection = np.add(self.red, self.green)/255
        decrement_area = self.decrement_area * self.red/255
        increment_area = self.increment_area * self.green/255
        anchor_area = self.anchor_area * self.array_2/255 # weighted by mass od second image (over the anchor area)
        anchor_decrement_area = self.anchor_decrement_area  * self.array_2/255 # weighted by mass od second image (over the anchor area)
        anchor_increment_area = self.anchor_increment_area * self.array_2/255 # weighted by mass od second image (over the anchor area)
                                
        self.data = {"decrement": {"data": self.red, "mass": str(np.sum(self.red)), "area": str(self.calculate_area_size(self.red, True)), "COG": self.calculate_cog(self.red, False), "COG_weighted": self.calculate_cog(self.red, True)},
                "increment": {"data": self.green,"mass": str(np.sum(self.green)), "area": str(self.calculate_area_size(self.green, True)), "COG": self.calculate_cog(self.green, False), "COG_weighted": self.calculate_cog(self.green, True)},
                #"anchor": {"data": self.blue, "COG": self.calculate_cog(self.blue, False), "COG_weighted": self.calculate_cog(self.blue, True)},
                "decrement_area": {"data": decrement_area, "mass": str(np.sum(decrement_area)), "area": str(self.calculate_area_size(decrement_area, True)), "COG": self.calculate_cog(decrement_area, False), "COG_weighted": self.calculate_cog(decrement_area, True)},
                "increment_area": {"data": increment_area,"mass": str(np.sum(increment_area)), "area": str(self.calculate_area_size(increment_area, True)), "COG": self.calculate_cog(increment_area, False), "COG_weighted": self.calculate_cog(increment_area, True)},
                "anchor_area": {"data": anchor_area, "mass": str(np.sum(anchor_area)), "area": str(self.calculate_area_size(anchor_area, True)), "COG": self.calculate_cog(anchor_area, False), "COG_weighted": self.calculate_cog(anchor_area, True)},
                "anchor_decrement_area": {"data": anchor_decrement_area, "mass": str(np.sum(anchor_decrement_area)), "area": str(self.calculate_area_size(anchor_decrement_area, True)), "COG": self.calculate_cog(anchor_decrement_area, False), "COG_weighted": self.calculate_cog(anchor_decrement_area, True)},
                "anchor_increment_area": {"data": anchor_increment_area, "mass": str(np.sum(anchor_increment_area)), "area": str(self.calculate_area_size(anchor_increment_area, True)), "COG": self.calculate_cog(anchor_increment_area, False), "COG_weighted": self.calculate_cog(anchor_increment_area, True)},
                "image_1": {"data": self.array_1, "mass": str(np.sum(self.array_1)), "area": str(self.calculate_area_size(self.array_1, True)), "COG": self.calculate_cog(self.array_1, False), "COG_weighted": self.calculate_cog(self.array_1, True)},
                "image_2": {"data": self.array_2, "mass": str(np.sum(self.array_2)), "area": str(self.calculate_area_size(self.array_2, True)), "COG": self.calculate_cog(self.array_2, False), "COG_weighted": self.calculate_cog(self.array_2, True)},
                }
        return self.data
        
    def get_AID_COG(self):
        """ Returns AID image in 8bit format with translocation vector.
        """
        pixel_size = np.sqrt(self.pixelArea)

        red = self.calculate_layer(self.red)
        green = self.calculate_layer(self.green)
        blue = self.calculate_layer(self.blue)

        if self.n_of_colors == 'two':
            AID = np.array([red, green, np.zeros_like(red)]).T
        if self.n_of_colors == 'four':
            AID = np.array([red, green, blue]).T
            
        self.AID_image_8b_COG = Image.fromarray(AID, 'RGB')
        
        red_COG = tuple([i/pixel_size for i in self.data['decrement']['COG_weighted']])
        green_COG = tuple([i/pixel_size for i in self.data['increment']['COG_weighted']])
                
        AID_image_8b_COG = ImageDraw.Draw(self.AID_image_8b_COG)
        AID_image_8b_COG.line((red_COG, green_COG), fill=(255, 255, 255), width=2)
        AID_image_8b_COG.pieslice(((red_COG[0]-3, red_COG[1]-3), (red_COG[0]+3, red_COG[1]+3)), 0, 360, fill=(0, 0, 255))
        AID_image_8b_COG.line(((red_COG[0], red_COG[1]), (red_COG[0]+20, red_COG[1])), fill=(0, 0, 255), width=2)

        return self.AID_image_8b_COG
    
    def get_images_COG(self):
        """ Returns a overlay of quantitative phase images in 8bit format with translocation vector.
        """
        pixel_size = np.sqrt(self.pixelArea)
        
        region = self.anchor_area + self.decrement_area + self.increment_area
        
        red = self.calculate_layer(self.array_1)
        red = np.where(region == 255, red, 0)
        blue = self.calculate_layer(self.array_2)
        blue = np.where(region == 255, blue, 0)

        image = np.array([red, np.zeros_like(red), blue]).T    
        
        self.compose_image_8b_COG = Image.fromarray(image, 'RGB')
        
        red_COG = tuple([i/pixel_size for i in self.data['image_1']['COG_weighted']])
        blue_COG = tuple([i/pixel_size for i in self.data['image_2']['COG_weighted']])
                
        compose_image_8b_COG = ImageDraw.Draw(self.compose_image_8b_COG)
        compose_image_8b_COG.line((red_COG, blue_COG), fill=(255, 255, 255), width=2)
        compose_image_8b_COG.pieslice(((red_COG[0]-3, red_COG[1]-3), (red_COG[0]+3, red_COG[1]+3)), 0, 360, fill=(0, 0, 255))
        compose_image_8b_COG.line(((red_COG[0], red_COG[1]), (red_COG[0]+20, red_COG[1])), fill=(0, 0, 255), width=2)
        
        return self.compose_image_8b_COG
        
class Loader(object):
    def __init__(self, path):
        self.path = path
        self.list_dir = []
        #self.filtred_dir = []
        self.max_diff = 0
        self.min_diff = 0
        
    def diff(self, array_1, array_2):
        """ Calculates a difference between two numpy arrays.
        """
        result = array_2 - array_1
        return result
    
    def read_directory(self):
        """ Reads a directory and loads tiff files.
        """
        for file in os.listdir(self.path):
            if file.endswith(".tif"):
                self.list_dir.append(file)
            if file.endswith(".tiff"):
                self.list_dir.append(file)    
        self.list_dir.sort()
            
    def map_directory(self):
        """ Returns maximal and minimal differences in the values of all series of images.
        """
        max_diff = []
        min_diff = []
        for file_1 in self.list_dir:
            array_1 = np.array(Image.open(os.path.join(self.path, file_1)))
            for file_2 in self.list_dir:
                if file_1 != file_2:
                    array_2 = np.array(Image.open(os.path.join(self.path, file_2)))         
                    diff = abs(self.diff(array_1, array_2))
                    max_diff.append(diff.max())
                    min_diff.append(diff.min())
        self.max_diff = max(max_diff)
        self.min_diff = min(min_diff)

"""
class Image_processing(object):
    def __init__(self, trashold_proc=0.3):
        self.trashold_proc = trashold_proc
            
    def get_contour(self, image, width_of_line=1):
        array = np.array(image)
        max_array = array.max()
        array = ((array/max_array)*255).astype(np.uint8)     
        ret, thresh = cv.threshold(array, 255*self.trashold_proc, max_array, 0)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        empty = np.zeros_like(array)
        return cv.drawContours(empty, contours, -1, (255,0,0), width_of_line)
    
    def image_and_contour(self, image):
        contour = self.get_contour(image)
        layer_1 = np.array(image)
        max_layer_1 = layer_1.max()
        layer_1 = ((layer_1/max_layer_1)*255).astype(np.uint8)
        nothing = np.zeros_like(layer_1)
        mix = np.array([layer_1, contour, nothing]).T
        image = Image.fromarray(mix, 'RGB')
        return image
"""

class Core(object):
    def __init__(self, path, trashold, quantitative, step, n_of_colors, mag, wavelength, rad_to_pg, AID_type):
        self.path = path
        self.trashold = trashold
        self.quantitative = quantitative
        self.step = step
        self.n_of_colors = n_of_colors
        self.mag = mag
        self.wavelength = wavelength
        self.rad_to_pg = rad_to_pg
        self.AID_type = AID_type
        
        self.list_dir = []
        self.list_dir_out = []
        self.max_diff = 0
        self.min_diff = 0
        self.AID = []
        self.AID_names = []
        self.zero_line_AID = []
        self.zero_lines = []
        self.zero_lines_len = []
        
        self.sorted_names_of_images = []
        self.sorted_images = []
        self.sorted_contours = []
        self.sorted_images_and_contours = []
        
        self.sorted_area_masks = []
        self.sorted_extended_anchor_areas = []
        
        self.sorted_COG = []
        self.sorted_AID_COG = []
        self.sorted_images_COG = []
        

    def reset_of_outputs(self):
        """ Reset all outputs.
        """
        self.sorted_names_of_images = []
        self.sorted_images = []
        self.sorted_contours = []
        self.sorted_images_and_contours = []

    def make_dir(self, path):
        """ Creates directory by the path input.
        """
        try: 
            os.mkdir(path)
        except OSError as error: 
            print(error) 
    
    def save_COG_data(self, list_item, path, category):
        """ For given category (list_item) saves data to numpy file and saves data to B&W image.
        """
        os.chdir(path)
        data = list_item.get(category).get("data")
        np.save(category+'.npy', data)
            
        data_8b = ((data/np.amax(data))*255).astype(np.uint8)
        image_8b = Image.fromarray(np.array([data_8b, data_8b, data_8b]).T, 'RGB')
        #image_8b_mirror = ImageOps.mirror(image_8b)
        image_8b.save(category+'.png')
        
    def save_JSON_COG(self, list_item, path, category):
        """ For given category (list_item) saves mass and centres of gravities to json file.
        """
        os.chdir(path)
        COGs = list_item.get(category)
        del COGs["data"]
        COGs_json = json.dumps(COGs)
        file = open(category+".json","w")
        file.write(COGs_json)
        file.close()    
        
    def rad_to_pg(self, array, mag='40x', wavelength=650):
        """ Converts array from radians to cell mass distribution.
        """
        CHI = 0.2022403133761694
        if mag == MAGNIFICATION[0]:
            pixelArea = PIXEL_AREA[0]
        if mag == MAGNIFICATION[1]:
            pixelArea = PIXEL_AREA[1]
        if mag == MAGNIFICATION[2]:
            pixelArea = PIXEL_AREA[2]
        if mag == MAGNIFICATION[3]:
            pixelArea = PIXEL_AREA[3]
        if mag == MAGNIFICATION[4]:
            pixelArea = PIXEL_AREA[4]

        aInCm2 = pixelArea*(10**-4)**2
        opd2PiCm = wavelength*10**-7 
        massScaleG = 1/(2*np.pi)*opd2PiCm*aInCm2/CHI
        massScalePG = massScaleG*10**12
        array_pg = array * massScalePG
        
        return array_pg
    
    def image_open(self, path):
        """ Opens image and convert data to numpy array.
        """
        image = Image.open(path) 
        array = np.array(image)
        array = np.rot90(array)
        array = np.flipud(array)
        return array
    
    def increment_AID(self, max_val):
        """ Provides incremental AID analysis.
        """
        for index in range(0, len(self.list_dir)-self.step, self.step):
            array_1 = self.image_open(os.path.join(self.path,self.list_dir[index] ))
            self.list_dir_out.append(self.list_dir[index])
            array_2 = self.image_open(os.path.join(self.path,self.list_dir[index+self.step] ))
            if self.rad_to_pg == 'True':
                array_1 = self.rad_to_pg(array_1, self.mag, self.wavelength)
                array_2 = self.rad_to_pg(array_2, self.mag, self.wavelength)
                 
                
            AID_class = AID(array_1, array_2, self.trashold, max_val, self.n_of_colors, self.mag, self.rad_to_pg)
            
            # AID
            AID_data = AID_class.get_AID_image()
            self.AID.append(AID_data)
            self.AID_names.append(self.list_dir[index+self.step] + '-' + self.list_dir[index])
            
            # zero_lines
            self.zero_line_AID.append(AID_class.get_zero_line_AID())
            self.zero_lines.append(AID_class.zero_line)
            self.zero_lines_len.append(AID_class.zero_line_len)

            # areas_masks
            
            self.sorted_area_masks.append(AID_class.get_area_masks())
            self.sorted_extended_anchor_areas.append(AID_class.get_extended_anchor_areas())
            
            # centers of gravity
            
            self.sorted_COG.append(AID_class.make_masked_part_of_AID())
            
            # AID center of gravity
            
            self.sorted_AID_COG.append(AID_class.get_AID_COG())
            
            # images COG
            
            self.sorted_images_COG.append(AID_class.get_images_COG())
            
            del AID_class
        
    def to_first_AID(self, max_val):
        """ Provides "To First" AID analysis.
        """
        for index in range(0, len(self.list_dir)-self.step, self.step):
            array_1 = self.image_open(os.path.join(self.path,self.list_dir[0] ))
            self.list_dir_out.append(self.list_dir[index])
            array_2 = self.image_open(os.path.join(self.path,self.list_dir[index+self.step] ))
            if self.rad_to_pg == 'True':
                array_1 = self.rad_to_pg(array_1, self.mag, self.wavelength)
                array_2 = self.rad_to_pg(array_2, self.mag, self.wavelength)
                 
                
            AID_class = AID(array_1, array_2, self.trashold, max_val, self.n_of_colors, self.mag, self.rad_to_pg)
            
            # AID
            AID_data = AID_class.get_AID_image()
            self.AID.append(AID_data)
            self.AID_names.append(self.list_dir[index+self.step] + '-' + self.list_dir[0])
            
            # zero_lines
            self.zero_line_AID.append(AID_class.get_zero_line_AID())
            self.zero_lines.append(AID_class.zero_line)
            self.zero_lines_len.append(AID_class.zero_line_len)

            # areas_masks
            
            self.sorted_area_masks.append(AID_class.get_area_masks())
            self.sorted_extended_anchor_areas.append(AID_class.get_extended_anchor_areas())
            
            # centers of gravity
            
            self.sorted_COG.append(AID_class.make_masked_part_of_AID())
            
            # AID center of gravity
            
            self.sorted_AID_COG.append(AID_class.get_AID_COG())
            
            # images COG
            
            self.sorted_images_COG.append(AID_class.get_images_COG())
            
            del AID_class
    
################################################################################################################        
    def generate_AIDs(self):
        """ Loads data fom files and runs AID analysis.
        """
        loader = Loader(self.path)
        loader.read_directory()
        loader.map_directory()
        self.list_dir = loader.list_dir
        self.max_diff = loader.max_diff
        self.min_diff = loader.min_diff
        del loader
        
        if self.quantitative == True:
            max_val = self.max_diff
        else:
            max_val = 1
        if self.AID_type == "increment":
            self.increment_AID(max_val)
        if self.AID_type == "to_first":
            self.to_first_AID(max_val)

                
    def save_data(self):
        """ Saves all data.
        """
        ########################### AID #########################################
        path_AID = os.path.join(self.path,"AID")
        self.make_dir(path_AID)
        path_AID_raw = os.path.join(self.path,"AID_raw")
        self.make_dir(path_AID_raw)
        for index, item in enumerate(self.AID):
            #item_mirror = ImageOps.mirror(item[0])
            item[0].save(os.path.join(path_AID, self.AID_names[index])+ '.png')
            np.save((os.path.join(path_AID_raw, self.AID_names[index])+'.npy'), item[1])
        
        ########################## AREA MASKS ###################################    
        area_masks = os.path.join(self.path,"area_masks")
        self.make_dir(area_masks)
        area_masks_raw = os.path.join(self.path,"area_masks_raw")
        self.make_dir(area_masks_raw)
        for index, item in enumerate(self.sorted_area_masks):
            #item_mirror = ImageOps.mirror(item[0])
            item[0].save(os.path.join(area_masks, self.AID_names[index])+ '.png')
            np.save((os.path.join(area_masks_raw, self.AID_names[index])+'.npy'), item[1])
        
        ########################### AREA MASKS EXTENDED #########################
        area_masks_extended = os.path.join(self.path,"area_masks_extended")
        self.make_dir(area_masks_extended)
        area_masks_extended_raw = os.path.join(self.path,"area_masks_extended_raw")
        self.make_dir(area_masks_extended_raw)
        for index, item in enumerate(self.sorted_extended_anchor_areas):
            #item_mirror = ImageOps.mirror(item[0])
            item[0].save(os.path.join(area_masks_extended, self.AID_names[index])+ '.png')
            np.save((os.path.join(area_masks_extended_raw, self.AID_names[index])+'.npy'), item[1])
        
        ########################### ZERO LINE AID ###############################
        zero_line_AID = os.path.join(self.path,"zero_line_AID")
        self.make_dir(zero_line_AID)
        zero_line_AID_raw = os.path.join(self.path,"zero_line_AID_raw")
        self.make_dir(zero_line_AID_raw)
        for index, item in enumerate(self.zero_line_AID):
            #item_mirror = ImageOps.mirror(item[0])
            item[0].save(os.path.join(zero_line_AID, self.AID_names[index])+ '.png')
            np.save((os.path.join(zero_line_AID_raw, self.AID_names[index])+'.npy'), item[1])
        
        ########################### DATA ####################################
        folder_name = os.path.join(self.path, "data")
        self.make_dir(folder_name)
        for index, list_item in enumerate(self.sorted_COG):
            name = self.AID_names[index]
            subfolder_name = os.path.join(folder_name, name)
            self.make_dir(subfolder_name)
            
            for category in list_item.keys():
                self.save_COG_data(list_item, subfolder_name, category)
                self.save_JSON_COG(list_item, subfolder_name, category)

            file = open("zeroline.json", "w")
            export = {"zero-line lenght": str(self.zero_lines_len[index])}
            file.write(json.dumps(export))
            file.close()

            ########################### AID COG #########################################
        path_AID_COG = os.path.join(self.path,"AID_COG")
        self.make_dir(path_AID_COG)
        for index, item in enumerate(self.sorted_AID_COG):
            #item_mirror = ImageOps.mirror(item)
            item.save(os.path.join(path_AID_COG, self.AID_names[index])+ '.png')
            
        ########################### image COG #########################################
        path_images_COG = os.path.join(self.path,"images_COG")
        self.make_dir(path_images_COG)
        for index, item in enumerate(self.sorted_images_COG):
            #item_mirror = ImageOps.mirror(item)
            item.save(os.path.join(path_images_COG, self.AID_names[index])+ '.png')
                
        ################################ SAVE FOR GRAFS ###########################
        path_export = os.path.join(self.path,"export")
        self.make_dir(path_export)
        np.save((os.path.join(path_export,'sorted_COG.npy')), self.sorted_COG)
        np.save((os.path.join(path_export,'AID_names.npy')), self.AID_names)
        np.save((os.path.join(path_export,'list_dir_out.npy')), self.list_dir_out)
        

######################################################################################################

           


    def generate_sorted_images(self):
        """ Sort files from directory with tiff files.
        """
        loader = Loader(self.path)
        loader.read_directory()
        loader.map_directory()
        self.list_dir = loader.list_dir
        self.max_diff = loader.max_diff
        self.min_diff = loader.min_diff
        del loader       
        
        self.reset_of_outputs()
        
        for index in range(0, len(self.list_dir), self.step):
            self.sorted_images.append(Image.open(os.path.join(self.path,self.list_dir[index] )))
            self.sorted_names_of_images.append(self.list_dir[index]) 
    """

    def get_images_and_contours(self):
        self.generate_sorted_images()
        image_processing = Image_processing(0)
        for image in self.sorted_images:
            self.sorted_contours.append(image_processing.get_contour(image))
            self.sorted_images_and_contours.append(image_processing.image_and_contour(image))
    """
        
if __name__ == "__main__":
   pass
