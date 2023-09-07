#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:41:03 2020

@author: enterprise
"""
import os

import numpy as np
import math

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams

import xlsxwriter 



class Grafs(object):
    def __init__(self, path, rad_to_pg):
        self.path_export = os.path.join(path,"export")
        self.sorted_COG = list(np.load(os.path.join(self.path_export, 'sorted_COG.npy'), allow_pickle=True) )
        self.AID_names = list(np.load(os.path.join(self.path_export, 'AID_names.npy'), allow_pickle=True) )
        self.list_dir_out = list(np.load(os.path.join(self.path_export, 'list_dir_out.npy'), allow_pickle=True) )
        self.rad_to_pg = rad_to_pg
        
        
    def plot_data(self, data1_x, data1_y, name1, data2_x, data2_y, name2, name):
        """ Plot graphs with arrows diagram and bar graphs. Category is given by data definition.
        """
        labels=self.AID_names
        
        d1_x = np.asarray(data1_x)
        d1_y = np.asarray(data1_y)
        d2_x = np.asarray(data2_x)
        d2_y = np.asarray(data2_y)
        angle, distance = self.measure(data1_x, data1_y)
        angle_w, distance_w = self.measure(data2_x, data2_y)
        
        fig, axs = plt.subplots(3, 2)
        plt.tight_layout()
        #plt.subplots_adjust(wspace = 0.2)

        p = axs[0,0].plot(d1_x, d1_y, 'bo', linestyle='-', markersize=3, alpha=0)
        for x,y,dx,dy in zip(d1_x[:-1], d1_y[:-1], d1_x[1:], d1_y[1:]):
           axs[0,0].annotate('', xy=(dx,dy),  xycoords='data',
                        xytext=(x,y), textcoords='data',
                        arrowprops=dict(arrowstyle="->, head_width=0.5, head_length = 1"))
                   
        for i, label in enumerate(labels):
           x_loc = d1_x[i]
           y_loc = d1_y[i]
           txt = axs[0,0].annotate(label, xy=(x_loc, y_loc), size=10,
               xytext=(-60, 10), textcoords='offset points',
               arrowprops=None)
        if self.rad_to_pg == True:
            axs[0,0].set_xlabel('x (um)')
            axs[0,0].set_ylabel('y (um)')
        else: 
            axs[0,1].set_xlabel('x (pixel)')
            axs[0,1].set_ylabel('y (pixel)')
        axs[0,0].set_title(name1, fontsize=12)
        
        ###############################################################################################
        
        q = axs[0,1].plot(d2_x, d2_y, 'bo', linestyle='-', markersize=3, alpha=0)
        for x,y,dx,dy in zip(d2_x[:-1], d2_y[:-1], d2_x[1:], d2_y[1:]):
           axs[0,1].annotate('', xy=(dx,dy),  xycoords='data',
                        xytext=(x,y), textcoords='data',
                        arrowprops=dict(arrowstyle="->, head_width=0.5, head_length = 1"))
                  
        for i, label in enumerate(labels):
           x_loc = d2_x[i]
           y_loc = d2_y[i]
           txt = axs[0,1].annotate(label, xy=(x_loc, y_loc), size=10,
               xytext=(-60, 10), textcoords='offset points',
               arrowprops=None)
        
        if self.rad_to_pg == True:
            axs[0,0].set_xlabel('x (um)')
            axs[0,0].set_ylabel('y (um)')
        else: 
            axs[0,1].set_xlabel('x (pixel)')
            axs[0,1].set_ylabel('y (pixel)')
        axs[0,1].set_title(name2, fontsize=12)
        
        ###############################################################################################

        axs[1,0].bar(self.AID_names[1:], distance)
        axs[1,0].set_title('COG', fontsize=12)
        if self.rad_to_pg == True:
            axs[1,0].set(xlabel='AID',  ylabel='displacement (um)')
        else:
            axs[1,0].set(xlabel='AID',  ylabel='displacement (pixel)')
        axs[1,0].set_xticklabels('')
        
        axs[2,0].bar(self.AID_names[1:], angle)
        axs[2,0].set_title('COG_angle', fontsize=12)
        axs[2,0].set(xlabel='AID', ylabel='angle (°)')
        
        axs[1,1].bar(self.AID_names[1:], distance_w)
        axs[1,1].set_title('COG_weighted', fontsize=12)
        if self.rad_to_pg == True:
            axs[1,1].set(xlabel='AID',  ylabel='displacement (um)')
        else:
            axs[1,1].set(xlabel='AID',  ylabel='displacement (pixel)')
        axs[1,1].set_xticklabels('')

        axs[2,1].bar(self.AID_names[1:], angle_w)
        axs[2,1].set_title('COG_weighted_angle', fontsize=12)
        axs[2,1].set(xlabel='AID', ylabel='angle (°)')
        
        for ax in fig.axes[2:6]:
            matplotlib.pyplot.sca(ax)
            plt.xticks(rotation=90)
        
        fig.set_size_inches(20, 30)
        plt.savefig(os.path.join(self.path_export, name1+' and '+name2+'.png'))
        plt.close(fig)

    def plot_green_red_difference_COG_data(self, decrement_COGw_x, decrement_COGw_y, increment_COGw_x, increment_COGw_y):
        """ Plot increment-decrement bar graphs.
        """
        angle_w = []
        distance_w = []
        
        for index in range(len(increment_COGw_x)):
            x_dist = (increment_COGw_x[index] - decrement_COGw_x[index])
            y_dist = (increment_COGw_y[index] - decrement_COGw_y[index])
            distance_w.append(math.sqrt(x_dist * x_dist + y_dist * y_dist))
            angle = -1*math.degrees(math.atan2(y_dist, x_dist))
            if angle < 0:                
                angle = angle + 360
            
            angle_w.append(angle)
            
        fig, axs = plt.subplots(1, 2)
        
        axs[1].bar(self.AID_names, angle_w)
        axs[1].set_title('increment-decrement_COG_angle', fontsize=12)
        axs[1].set(xlabel='AID', ylabel='angle (°)')
        
        axs[0].bar(self.AID_names, distance_w)
        axs[0].set_title('increment-decrement_COG_displacement', fontsize=12)
        if self.rad_to_pg == True:
            axs[0].set(xlabel='AID', ylabel='displacement (um)')
        else:
            axs[0].set(xlabel='AID', ylabel='displacement (pixel)')

        for ax in fig.axes:
            matplotlib.pyplot.sca(ax)
            plt.xticks(rotation=90)
        
        fig.set_size_inches(20, 10)
        rcParams.update({'figure.autolayout': True})
        plt.savefig(os.path.join(self.path_export, 'increment-decrement_weighted.png'))
        plt.close(fig) 
        
        return distance_w, angle_w

        
    def measure(self, x, y):
        """ Returns distances and angles between points given by lists x and y.
        """
        angle = []
        distance = []
        for index in range(len(x)-1):
            x_dist = (x[index+1] - x[index])
            y_dist = (y[index+1] - y[index])
            distance.append(math.sqrt(x_dist * x_dist + y_dist * y_dist))
            ang = math.degrees(math.atan2(y_dist, x_dist))
            if ang < 0:
                ang = ang + 360
            angle.append(ang)
        
        return np.asarray(angle), np.asarray(distance)
    
    def plot_mass(self, image_1_mass, image_2_mass, increment_mass, decrement_mass, list_dir_out):
        """ Creates bar graphs of dry mass of the cell, of increment of the cell and of decrement of the cell
        """
        fig, axs = plt.subplots(2, 2)

        axs[0,0].bar(list_dir_out, image_1_mass)
        axs[0,0].set_title('image x mass', fontsize=12)
        axs[0,0].set(xlabel='image', ylabel='mass (pg)')
        
        axs[0,1].bar(self.AID_names, increment_mass)
        axs[0,1].set_title('increment x mass', fontsize=12)
        axs[0,1].set(xlabel='AID', ylabel='mass (pg)')
        
        axs[1,0].bar(self.AID_names, decrement_mass)
        axs[1,0].set_title('decrement x mass', fontsize=12)
        axs[1,0].set(xlabel='AID', ylabel='mass (pg)')
        
        axs[1, 1].remove()
        
        
        for ax in fig.axes:
            matplotlib.pyplot.sca(ax)
            plt.xticks(rotation=90)
        
        fig.set_size_inches(20, 20)
        rcParams.update({'figure.autolayout': True})
        plt.savefig(os.path.join(self.path_export, 'mass.png'))
        plt.close(fig) 
            

        
        return image_1_mass

    def plot_all(self, data1_x, data1_y, name1, data1w_x, data1w_y, name1w, \
                       data2_x, data2_y, name2, data2w_x, data2w_y, name2w, \
                       data3_x, data3_y, name3, data3w_x, data3w_y, name3w, \
                       data4_x, data4_y, name4, data4w_x, data4w_y, name4w, \
                       data5_x, data5_y, name5, data5w_x, data5w_y, name5w, \
                       data6_x, data6_y, name6, data6w_x, data6w_y, name6w, \
                       data7_x, data7_y, name7, data7w_x, data7w_y, name7w, \
                       data8_x, data8_y, name8, data8w_x, data8w_y, name8w, \
                       data9_x, data9_y, name9, data9w_x, data9w_y, name9w, \
                       distance10w, angle10w, name10w):
        """ Plots all data values to the graphs.
        """

        angle1, distance1 = self.measure(data1_x, data1_y)
        angle1w, distance1w = self.measure(data1w_x, data1w_y)
        angle2, distance2 = self.measure(data2_x, data2_y)
        angle2w, distance2w = self.measure(data2w_x, data2w_y)
        angle3, distance3 = self.measure(data3_x, data3_y)
        angle3w, distance3w = self.measure(data3w_x, data3w_y)
        angle4, distance4 = self.measure(data4_x, data4_y)
        angle4w, distance4w = self.measure(data4w_x, data4w_y)
        angle5, distance5 = self.measure(data5_x, data5_y)
        angle5w, distance5w = self.measure(data5w_x, data5w_y)
        angle6, distance6 = self.measure(data6_x, data6_y)
        angle6w, distance6w = self.measure(data6w_x, data6w_y)
        angle7, distance7 = self.measure(data7_x, data7_y)
        angle7w, distance7w = self.measure(data7w_x, data7w_y)
        angle8, distance8 = self.measure(data8_x, data8_y)
        angle8w, distance8w = self.measure(data8w_x, data8w_y)
        angle9, distance9 = self.measure(data9_x, data9_y)
        angle9w, distance9w = self.measure(data9w_x, data9w_y)


        fig, axs = plt.subplots(2, 2)
        plt.tight_layout()

        axs[0, 0].plot(self.AID_names[1:], distance1, '-o', label = name1)
        axs[0, 0].plot(self.AID_names[1:], distance2, '-o', label = name2)
        axs[0, 0].plot(self.AID_names[1:], distance3, '-o', label = name3)
        axs[0, 0].plot(self.AID_names[1:], distance4, '-o', label = name4)
        axs[0, 0].plot(self.AID_names[1:], distance5, '-o', label = name5)
        axs[0, 0].plot(self.AID_names[1:], distance6, '-o', label = name6)
        axs[0, 0].plot(self.AID_names[1:], distance7, '-o', label = name7)
        axs[0, 0].plot(self.AID_names[1:], distance8, '-o', label = name8)
        axs[0, 0].plot(self.AID_names[1:], distance9, '-o', label = name9)
        axs[0, 0].set_title('COG', fontsize=12)
        if self.rad_to_pg == True:
            axs[0, 0].set(xlabel='AID', ylabel='displacement (um)')
        else:
            axs[0, 0].set(xlabel='AID', ylabel='displacement (pixel)')
        axs[0, 0].legend()
        #axs[0, 0].set_xticklabels('')

        axs[0, 1].plot(self.AID_names[1:], angle1, '-o', label = name1)
        axs[0, 1].plot(self.AID_names[1:], angle2, '-o', label = name2)
        axs[0, 1].plot(self.AID_names[1:], angle3, '-o', label = name3)
        axs[0, 1].plot(self.AID_names[1:], angle4, '-o', label = name4)
        axs[0, 1].plot(self.AID_names[1:], angle5, '-o', label = name5)
        axs[0, 1].plot(self.AID_names[1:], angle6, '-o', label = name6)
        axs[0, 1].plot(self.AID_names[1:], angle7, '-o', label = name7)
        axs[0, 1].plot(self.AID_names[1:], angle8, '-o', label = name8)
        axs[0, 1].plot(self.AID_names[1:], angle9, '-o', label = name9)
        axs[0, 1].set_title('COG_angle', fontsize=12)
        axs[0, 1].set(xlabel='AID', ylabel='angle (°)')
        axs[0, 1].legend()

        axs[1, 0].plot(self.AID_names, distance10w, '-o', label = name10w)
        axs[1, 0].plot(self.AID_names[1:], distance1w, '-o', label = name1w)
        axs[1, 0].plot(self.AID_names[1:], distance2w, '-o', label = name2w)
        axs[1, 0].plot(self.AID_names[1:], distance3w, '-o', label = name3w)
        axs[1, 0].plot(self.AID_names[1:], distance4w, '-o', label = name4w)
        axs[1, 0].plot(self.AID_names[1:], distance5w, '-o', label = name5w)
        axs[1, 0].plot(self.AID_names[1:], distance6w, '-o', label = name6w)
        axs[1, 0].plot(self.AID_names[1:], distance7w, '-o', label = name7w)
        axs[1, 0].plot(self.AID_names[1:], distance8w, '-o', label = name8w)
        axs[1, 0].plot(self.AID_names[1:], distance9w, '-o', label = name9w)
        axs[1, 0].set_title('COG_weighted', fontsize=12)
        if self.rad_to_pg == True:
            axs[1, 0].set(xlabel='AID', ylabel='displacement (um)')
        else:
            axs[1, 0].set(xlabel='AID', ylabel='displacement (pixel)')
        axs[1, 0].legend()
        #axs[1, 0].set_xticklabels('')
        
        axs[1, 1].plot(self.AID_names, angle10w, '-o', label = name10w)
        axs[1, 1].plot(self.AID_names[1:], angle1w, '-o', label = name1w)
        axs[1, 1].plot(self.AID_names[1:], angle2w, '-o', label = name2w)
        axs[1, 1].plot(self.AID_names[1:], angle3w, '-o', label = name3w)
        axs[1, 1].plot(self.AID_names[1:], angle4w, '-o', label = name4w)
        axs[1, 1].plot(self.AID_names[1:], angle5w, '-o', label = name5w)
        axs[1, 1].plot(self.AID_names[1:], angle6w, '-o', label = name6w)
        axs[1, 1].plot(self.AID_names[1:], angle7w, '-o', label = name7w)
        axs[1, 1].plot(self.AID_names[1:], angle8w, '-o', label = name8w)
        axs[1, 1].plot(self.AID_names[1:], angle9w, '-o', label = name9w)
        axs[1, 1].set_title('COG_weighted_angle', fontsize=12)
        axs[1, 1].set(xlabel='AID', ylabel='displacement (°)')
        axs[1, 1].legend()

        for ax in fig.axes[0:4]:
            matplotlib.pyplot.sca(ax)
            plt.xticks(rotation=90)

        fig.set_size_inches(20, 20)
        plt.savefig(os.path.join(self.path_export, 'all'))
        plt.close(fig)
       
    def generate_export(self):
        """ Loads all data from AID export, generates all graphs and crates Excel file with all values.
        """

        anchor_area_COG_x = []
        anchor_area_COG_y = []
        anchor_increment_area_COG_x = []
        anchor_increment_area_COG_y = []
        anchor_decrement_area_COG_x = []
        anchor_decrement_area_COG_y = []   
        image_1_COG_x = []
        image_1_COG_y = []
        image_2_COG_x = []
        image_2_COG_y = []    
        increment_COG_x = []
        increment_COG_y = []
        increment_area_COG_x = []
        increment_area_COG_y = []
        decrement_COG_x = []
        decrement_COG_y = []
        decrement_area_COG_x = []
        decrement_area_COG_y = []
        
        anchor_area_COGw_x = []
        anchor_area_COGw_y = []
        anchor_increment_area_COGw_x = []
        anchor_increment_area_COGw_y = []
        anchor_decrement_area_COGw_x = []
        anchor_decrement_area_COGw_y = []   
        image_1_COGw_x = []
        image_1_COGw_y = []
        image_2_COGw_x = []
        image_2_COGw_y = []    
        increment_COGw_x = []
        increment_COGw_y = []
        increment_area_COGw_x = []
        increment_area_COGw_y = []
        decrement_COGw_x = []
        decrement_COGw_y = []
        decrement_area_COGw_x = []
        decrement_area_COGw_y = []
        
        image_1_mass = []
        image_2_mass = []
        increment_mass = []
        decrement_mass = []
        
        for index, list_item in enumerate(self.sorted_COG):
            anchor_area_COG_x.append(list_item['anchor_area']['COG'][0])
            anchor_area_COG_y.append(list_item['anchor_area']['COG'][1])   
            
            anchor_increment_area_COG_x.append(list_item['anchor_increment_area']['COG'][0])
            anchor_increment_area_COG_y.append(list_item['anchor_increment_area']['COG'][1])
           
            anchor_decrement_area_COG_x.append(list_item['anchor_decrement_area']['COG'][0])
            anchor_decrement_area_COG_y.append(list_item['anchor_decrement_area']['COG'][1])   
           
            image_1_COG_x.append(list_item['image_1']['COG'][0])
            image_1_COG_y.append(list_item['image_1']['COG'][1])  
           
            image_2_COG_x.append(list_item['image_2']['COG'][0])
            image_2_COG_y.append(list_item['image_2']['COG'][1])  
           
            increment_COG_x.append(list_item['increment']['COG'][0])
            increment_COG_y.append(list_item['increment']['COG'][1])
                      
            increment_area_COG_x.append(list_item['increment_area']['COG'][0])
            increment_area_COG_y.append(list_item['increment_area']['COG'][1])
                        
            decrement_COG_x.append(list_item['decrement']['COG'][0])
            decrement_COG_y.append(list_item['decrement']['COG'][1])
                        
            decrement_area_COG_x.append(list_item['decrement_area']['COG'][0])
            decrement_area_COG_y.append(list_item['decrement_area']['COG'][1])
           
            #########################################################################################
            anchor_area_COGw_x.append(list_item['anchor_area']['COG_weighted'][0])
            anchor_area_COGw_y.append(list_item['anchor_area']['COG_weighted'][1])   
           
            anchor_increment_area_COGw_x.append(list_item['anchor_increment_area']['COG_weighted'][0])
            anchor_increment_area_COGw_y.append(list_item['anchor_increment_area']['COG_weighted'][1])
           
            anchor_decrement_area_COGw_x.append(list_item['anchor_decrement_area']['COG_weighted'][0])
            anchor_decrement_area_COGw_y.append(list_item['anchor_decrement_area']['COG_weighted'][1])   
           
            image_1_COGw_x.append(list_item['image_1']['COG_weighted'][0])
            image_1_COGw_y.append(list_item['image_1']['COG_weighted'][1])  
           
            image_2_COGw_x.append(list_item['image_2']['COG_weighted'][0])
            image_2_COGw_y.append(list_item['image_2']['COG_weighted'][1])  
           
            increment_COGw_x.append(list_item['increment']['COG_weighted'][0])
            increment_COGw_y.append(list_item['increment']['COG_weighted'][1])
                      
            increment_area_COGw_x.append(list_item['increment_area']['COG_weighted'][0])
            increment_area_COGw_y.append(list_item['increment_area']['COG_weighted'][1])
                        
            decrement_COGw_x.append(list_item['decrement']['COG_weighted'][0])
            decrement_COGw_y.append(list_item['decrement']['COG_weighted'][1])
                        
            decrement_area_COGw_x.append(list_item['decrement_area']['COG_weighted'][0])
            decrement_area_COGw_y.append(list_item['decrement_area']['COG_weighted'][1])
            ########################################################################################
            
            image_1_mass.append(float(list_item['image_1']['mass']))
            image_2_mass.append(float(list_item['image_2']['mass']))
            increment_mass.append(float(list_item['increment']['mass']))
            decrement_mass.append(float(list_item['decrement']['mass']))
                 
        self.plot_data(anchor_area_COG_x, anchor_area_COG_y, 'anchor_area_COG', anchor_area_COGw_x, anchor_area_COGw_y, 'anchor_area_COG_weighted','anchor_area')
        self.plot_data(anchor_increment_area_COG_x, anchor_increment_area_COG_y, 'anchor_increment_area_COG', anchor_increment_area_COGw_x, anchor_increment_area_COGw_y, 'anchor_increment_area_COG_weighted', 'anchor_increment_area')
        self.plot_data(anchor_decrement_area_COG_x, anchor_decrement_area_COG_y, 'anchor_decrement_area_COG', anchor_decrement_area_COGw_x, anchor_decrement_area_COGw_y, 'anchor_decrement_area_COG_weighted', 'anchor_decrement_area')
        self.plot_data(image_1_COG_x, image_1_COG_y, 'image_1_COG', image_1_COGw_x, image_1_COGw_y, 'image_1_COG_weighted', 'image_1')
        self.plot_data(image_2_COG_x, image_2_COG_y, 'image_2_COG', image_2_COGw_x, image_2_COGw_y, 'image_2_COG_weighted','image_2')
        self.plot_data(increment_COG_x, increment_COG_y, 'increment_COG', increment_COGw_x, increment_COGw_y, 'increment_COG_weighted',  'increment')
        self.plot_data(increment_area_COG_x, increment_area_COG_y, 'increment_area_COG', increment_area_COGw_x, increment_area_COGw_y, 'increment_area_COG_weighted', 'increment_area')
        self.plot_data(decrement_COG_x, decrement_COG_y, 'decrement_COG', decrement_COGw_x, decrement_COGw_y, 'decrement_COG_weighted', 'decrement')
        self.plot_data(decrement_area_COG_x, decrement_area_COG_y, 'decrement_area_COG', decrement_area_COGw_x, decrement_area_COGw_y, 'decrement_area_COG_weighted', 'decrement_area')
    
        green_red_difference_COG_distance_w, green_red_difference_COG_angle_w = self.plot_green_red_difference_COG_data(decrement_COGw_x, decrement_COGw_y, increment_COGw_x, increment_COGw_y)

        self.plot_all(image_1_COG_x, image_1_COG_y, 'image_1_COG', image_1_COGw_x, image_1_COGw_y, 'image_1_COG_weighted',
                      image_2_COG_x, image_2_COG_y, 'image_2_COG', image_2_COGw_x, image_2_COGw_y, 'image_2_COG_weighted',
                      increment_COG_x, increment_COG_y, 'increment_COG', increment_COGw_x, increment_COGw_y, 'increment_COG_weighted',
                      increment_area_COG_x, increment_area_COG_y, 'increment_area_COG', increment_area_COGw_x, increment_area_COGw_y, 'increment_area_COG_weighted',
                      decrement_COG_x, decrement_COG_y, 'decrement_COG', decrement_COGw_x, decrement_COGw_y, 'decrement_COG_weighted',
                      decrement_area_COG_x, decrement_area_COG_y, 'decrement_area_COG', decrement_area_COGw_x, decrement_area_COGw_y, 'decrement_area_COG_weighted',
                      anchor_area_COG_x, anchor_area_COG_y, 'anchor_area_COG', anchor_area_COGw_x, anchor_area_COGw_y, 'anchor_area_COG_weighted',
                      anchor_increment_area_COG_x, anchor_increment_area_COG_y, 'anchor_increment_area_COG', anchor_increment_area_COGw_x, anchor_increment_area_COGw_y, 'anchor_increment_area_COG_weighted',
                      anchor_decrement_area_COG_x, anchor_decrement_area_COG_y, 'anchor_decrement_area_COG', anchor_decrement_area_COGw_x, anchor_decrement_area_COGw_y, 'anchor_decrement_area_COG_weighted',
                      green_red_difference_COG_distance_w, green_red_difference_COG_angle_w, 'increment-decrement_weighted')

        all_image_data = self.plot_mass(image_1_mass, image_2_mass, increment_mass, decrement_mass, self.list_dir_out)
        
        
        workbook = xlsxwriter.Workbook(os.path.join(self.path_export, 'export_data.xlsx')) 
        bold = workbook.add_format({'bold': True})


        #######################################################################################################
        titles = self.list_dir_out
        data = all_image_data

        worksheet = workbook.add_worksheet('mass_images') 
        
        row = 0
        for index in range(len(titles)):
            column = 0
            worksheet.write(row, column, titles[index], bold)
            column = 1
            worksheet.write(row, column, all_image_data[index]) 
            column += 1
            row += 1
            
        #######################################################################################################
        titles = ['AID_name', 'increment_mass', 'decrement_mass']
        data = [self.AID_names, increment_mass, decrement_mass]

        worksheet = workbook.add_worksheet('mass_AID')
        
        row = 0
        for index in range(len(titles)):
            column = 0
            worksheet.write(row, column, titles[index], bold)
            column = 1
            for item in data[index] : 
                worksheet.write(row, column, item) 
                column += 1
            row += 1
         
        #######################################################################################################
        titles = ['AID_name', 'image_1_COG_x', 'image_1_COG_y',
                  'image_2_COG_x', 'image_2_COG_y', 
                  'anchor_area_COG_x', 'anchor_area_COG_y', 'anchor_increment_area_COG_x', 'anchor_increment_area_COG_y', 'anchor_decrement_area_COG_x', 'anchor_decrement_area_COG_y',
                  'increment_COG_x', 'increment_COG_y', 'increment_area_COG_x', 'increment_area_COG_y',
                  'decrement_COG_x', 'decrement_COG_y', 'decrement_area_COG_x', 'decrement_area_COG_y']
        data = [self.AID_names, image_1_COG_x, image_1_COG_y,
                  image_2_COG_x, image_2_COG_y, 
                  anchor_area_COG_x, anchor_area_COG_y, anchor_increment_area_COG_x, anchor_increment_area_COG_y, anchor_decrement_area_COG_x, anchor_decrement_area_COG_y,
                  increment_COG_x, increment_COG_y, increment_area_COG_x, increment_area_COG_y,
                  decrement_COG_x, decrement_COG_y, decrement_area_COG_x, decrement_area_COG_y]
        
        worksheet = workbook.add_worksheet('COG') 
                
        row = 0
        for index in range(len(titles)):
            column = 0
            worksheet.write(row, column, titles[index], bold)
            column = 1
            for item in data[index] : 
                worksheet.write(row, column, item) 
                column += 1
            row += 1
        #####################################################################################################
        titles = ['AID_name', 'image_1_COG_weighted_x', 'image_1_COG_weighted_y',
                  'image_2_COG_weighted_x', 'image_2_COG_weighted_y', 
                  'anchor_area_COG_weighted_x', 'anchor_area_COG_weighted_y', 'anchor_increment_area_COG_weighted_x', 'anchor_increment_area_COG_weighted_y', 'anchor_decrement_area_COG_weighted_x', 'anchor_decrement_area_COG_weighted_y',
                  'increment_COG_weighted_x', 'increment_COG_weighted_y', 'increment_area_COG_weighted_x', 'increment_area_COG_weighted_y',
                  'decrement_COG_weighted_x', 'decrement_COG_weighted_y', 'decrement_area_COG_weighted_x', 'decrement_area_COG_weighted_y']
        data = [self.AID_names, image_1_COGw_x, image_1_COGw_y,
                  image_2_COGw_x, image_2_COGw_y, 
                  anchor_area_COGw_x, anchor_area_COGw_y, anchor_increment_area_COGw_x, anchor_increment_area_COGw_y, anchor_decrement_area_COGw_x, anchor_decrement_area_COGw_y,
                  increment_COGw_x, increment_COGw_y, increment_area_COGw_x, increment_area_COGw_y,
                  decrement_COGw_x, decrement_COGw_y, decrement_area_COGw_x, decrement_area_COGw_y]
        
        worksheet = workbook.add_worksheet('COG_weighted') 
                
        row = 0
        for index in range(len(titles)):
            column = 0
            worksheet.write(row, column, titles[index], bold)
            column = 1
            for item in data[index] : 
                worksheet.write(row, column, item) 
                column += 1
            row += 1   
        #####################################################################################################
        titles = ['AID_name', 'increment-decrement_COG_displacement', 'increment-decrement_COG_angle']
        data = [self.AID_names, green_red_difference_COG_distance_w, green_red_difference_COG_angle_w]
        
        worksheet = workbook.add_worksheet('inc-dec_COG_weighted')
                
        row = 0
        for index in range(len(titles)):
            column = 0
            worksheet.write(row, column, titles[index], bold)
            column = 1
            for item in data[index] : 
                worksheet.write(row, column, item) 
                column += 1
            row += 1   
        
        
        workbook.close() 

        
        
        
if __name__ == "__main__":
    pass