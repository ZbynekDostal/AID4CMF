#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 28 00:37:30 2021

@author: defiant
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rcParams
import math
import xlsxwriter 

class Force(object):
    def __init__(self, path, rad_to_pg, time_step):
        self.path_export = os.path.join(path,"export")
        self.sorted_COG = list(np.load(os.path.join(self.path_export, 'sorted_COG.npy'), allow_pickle=True) )
        self.AID_names = list(np.load(os.path.join(self.path_export, 'AID_names.npy'), allow_pickle=True) )
        self.list_dir_out = list(np.load(os.path.join(self.path_export, 'list_dir_out.npy'), allow_pickle=True) )
        self.rad_to_pg = rad_to_pg
        self.time_step = time_step
        self.image_1_COGw = []
        self.image_2_COGw = []
        self.increment_COGw = []
        self.decrement_COGw = []
        self.image_1_mass = []
        self.image_2_mass = []
        self.increment_mass = []
        self.decrement_mass = []
        self.mass_pr = []
        self.momentum_vector_AID = []
        self.velocity_vector_AID = []
        self.displacement_vector_AID = []
        self.force_vector_AID = []
        self.angle = []
        self.sizes = []
        self.momentum_vector_AID_abs = []
        self.velocity_vector_AID_abs = []
        self.displacement_vector_AID_abs = []
        self.displacement_vector_AID_cumulative = []

        for index, list_item in enumerate(self.sorted_COG):
            self.image_1_COGw.append([(list_item['image_1']['COG_weighted'][0]), (list_item['image_1']['COG_weighted'][1])])  
            self.image_2_COGw.append([(list_item['image_2']['COG_weighted'][0]), (list_item['image_2']['COG_weighted'][1])])  
            self.increment_COGw.append([(list_item['increment']['COG_weighted'][0]), (list_item['increment']['COG_weighted'][1])])
            self.decrement_COGw.append([(list_item['decrement']['COG_weighted'][0]), (list_item['decrement']['COG_weighted'][1])])
            self.image_1_mass.append(float(list_item['image_1']['mass']))
            self.image_2_mass.append(float(list_item['image_2']['mass']))
            self.increment_mass.append(float(list_item['increment']['mass']))
            self.decrement_mass.append(float(list_item['decrement']['mass']))
        
    def mass_avr(self):
         """ Compensation of cell mass fluctuation due to cell recording inaccuracies.
         It can only be used for short observations if growth and loss of cell mass are neglected.
         """
         self.mass = sum(self.image_1_mass)/len(self.image_1_mass)
         i = 0
         for data in self.increment_mass:
             percent = data/self.image_1_mass[i]
             self.increment_mass[i] = percent * self.mass
             i = i + 1
         i = 0
         for data in self.decrement_mass:
             percent = data/self.image_1_mass[i]
             self.decrement_mass[i] = percent * self.mass
             i = i + 1
    
    def displacement_vector(self, start_point, end_point):
        """ Calculates displacement vectors in meters.
        """
        displacement_x = -(end_point[0] - start_point[0])*10**(-6)
        displacement_y = (end_point[1] - start_point[1])*10**(-6)
        return([displacement_x, displacement_y])
    
    def velocity_vector(self, start_point, end_point): 
        """ Calculates velocity vectors in meters/second.
        """
        displacement = self.displacement_vector(start_point, end_point)
        velocity_x = displacement[0]/self.time_step
        velocity_y = displacement[1]/self.time_step
        return([velocity_x, velocity_y], displacement)
        
    def momentum_vector(self, start_point, end_point, mass):
        """ Calculates momentum vectors in kilogram*meters/second.
        """
        velocity, displacement = self.velocity_vector(start_point, end_point)
        momentum_x = velocity[0]*mass*10**(-15)
        momentum_y = velocity[1]*mass*10**(-15)
        return([momentum_x, momentum_y], velocity, displacement)
        
    def calculate_force_AID(self):
        """ Calculates the subcellular Mechanical Force vectors in newtons.
        """
        for i in range(len(self.increment_mass)):
            mass = (self.increment_mass[i] + self.decrement_mass[i])/2

            self.mass_pr.append(mass*10**(-15))
            momentum_vector_AID, velocity_vector_AID, displacement_vector_AID = self.momentum_vector(self.increment_COGw[i], self.decrement_COGw[i], mass)
            self.displacement_vector_AID.append(displacement_vector_AID)
            self.velocity_vector_AID.append(velocity_vector_AID)
            self.momentum_vector_AID.append(momentum_vector_AID)
            self.displacement_vector_AID_abs.append( np.sqrt(displacement_vector_AID[0]**2 + displacement_vector_AID[1]**2) )
            if i == 0:
                self.displacement_vector_AID_cumulative.append( self.displacement_vector_AID_abs[i] )
            else:
                self.displacement_vector_AID_cumulative.append( self.displacement_vector_AID_abs[i] + self.displacement_vector_AID_cumulative[i-1] )
            self.velocity_vector_AID_abs.append( np.sqrt(velocity_vector_AID[0]**2 + velocity_vector_AID[1]**2) )            
            self.momentum_vector_AID_abs.append( np.sqrt(momentum_vector_AID[0]**2 + momentum_vector_AID[1]**2) )

        for i in range(len(self.increment_mass)):
            if i == 0:
                pass
            else:
                force_x = (self.momentum_vector_AID[i][0]-self.momentum_vector_AID[i-1][0])/self.time_step
                force_y = (self.momentum_vector_AID[i][1]-self.momentum_vector_AID[i-1][1])/self.time_step
                self.force_vector_AID.append([force_x, force_y])
    
    def measure(self, x, y):
        """ Returns distances and angles between points given by lists x and y.
        """
        angles = []
        sizes =  []
        for index in range(len(x)):
            ang = math.degrees(math.atan2(y[index], x[index]))
            size = np.sqrt(y[index]*y[index]+x[index]*x[index])
            if ang < 0:
                ang = ang + 360
                
            angles.append(ang)
            sizes.append(size)
        
        return np.asarray(angles), np.asarray(sizes)
            
    def plot_force(self):
        """ Creates rose plot of the forces and creates bar graphs of angles of forces and their components.
        """
        labels=self.AID_names

        f_x = np.array(self.force_vector_AID)[:,0]
        zero = np.zeros_like(f_x)
        f_x = f_x.tolist()
        f_y = (np.array(self.force_vector_AID)[:,1]).tolist()
        self.angle, self.sizes = self.measure(f_x, f_y)
        
        fig, axs = plt.subplots(2, 2)
        #plt.tight_layout()
        #plt.subplots_adjust(wspace = 0.2)

        p = axs[0,0].plot(f_x, f_y, 'bo', linestyle='-', markersize=3, alpha=0)
        axs[0,0].axis('equal')
        for x,y,dx,dy in zip(zero[:-1], zero[:-1], f_x[1:], f_y[1:]):
           axs[0,0].annotate('', xy=(dx,dy),  xycoords='data',
                        xytext=(x,y), textcoords='data',
                        arrowprops=dict(arrowstyle="->, head_width=0.5, head_length = 1"))
        
    
        for i, label in enumerate(labels[1:]):
           x_loc = f_x[i]
           y_loc = f_y[i]
           txt = axs[0,0].annotate(label, xy=(x_loc, y_loc), size=10,
               xytext=(-60, 10), textcoords='offset points',
               arrowprops=None)
        if self.rad_to_pg == True:
            axs[0,0].set_xlabel('size Fx (N)')
            axs[0,0].set_ylabel('size Fy (N)')
        else: 
            pass
        axs[0,0].set_title('Rose plot', fontsize=12)        
        
###############################################################################################

        axs[0,1].bar(self.AID_names[1:], f_x)
        axs[0,1].set_title('size Fx (N)', fontsize=12)
        if self.rad_to_pg == True:
            axs[0,1].set(xlabel='AID',  ylabel='size Fx (N)')
        else:
            pass
        axs[0,1].set_xticklabels('')
        
        axs[1,1].bar(self.AID_names[1:], f_y)
        axs[1,1].set_title('size Fy (N)', fontsize=12)
        if self.rad_to_pg == True:
            axs[1,1].set(xlabel='AID', ylabel='size Fy (N)')
        else:
            pass    
        
        axs[1,0].bar(self.AID_names[1:], self.angle)
        axs[1,0].set_title('F angle', fontsize=12)
        axs[1,0].set(xlabel='AID', ylabel='angle (°)')
        
        for ax in fig.axes[2:6]:
            matplotlib.pyplot.sca(ax)
            plt.xticks(rotation=90)        
        
        fig.set_size_inches(20, 20)
        plt.savefig(os.path.join(self.path_export, 'MF_graph.png'))
        plt.close(fig)
        
    def generate_xls(self):
        """ Generates Excel file with numerical data.
        """
        workbook = xlsxwriter.Workbook(os.path.join(self.path_export, 'export_MF.xlsx'))
        bold = workbook.add_format({'bold': True})
        
        f_x = np.array(self.force_vector_AID)[:,0]
        f_x = np.insert(f_x, 0, 0)
        f_y = np.array(self.force_vector_AID)[:,1]
        f_y = np.insert(f_y, 0, 0)
        self.sizes = np.insert(np.array(self.sizes), 0, 0)
        self.angle = np.insert(np.array(self.angle), 0, 0)

        titles = ['AID_name', 'mass [kg]', 'displacement_x [m]', 'displacement_y [m]', 'displacement_abs [m]', 'displacement_cumulative [m]', 'velocity_x [m/s]', 'velocity_y [m/s]', 'velocity_abs [m/s]', 'momentum_x [kg*m/s]', 'momentum_y [kg*m/s]', 'momentum_abs [kg*m/s]', 'force_x [N]', 'force_y [N]', 'force [N]', 'angle [°]']
        data = [self.AID_names, self.mass_pr,
                (np.array(self.displacement_vector_AID)[:,0]).tolist(), (np.array(self.displacement_vector_AID)[:,1]).tolist(), 
                np.array(self.displacement_vector_AID_abs), np.array(self.displacement_vector_AID_cumulative), 
                (np.array(self.velocity_vector_AID)[:,0]).tolist(), (np.array(self.velocity_vector_AID)[:,1]).tolist(), np.array(self.velocity_vector_AID_abs),
                (np.array(self.momentum_vector_AID)[:,0]).tolist(), (np.array(self.momentum_vector_AID)[:,1]).tolist(), np.array(self.momentum_vector_AID_abs),
                f_x.tolist(), f_y.tolist(), self.sizes.tolist(), self.angle.tolist()
                ] 

        worksheet = workbook.add_worksheet('MF')
        
        row = 0
        for index in range(len(titles)):
            column = 0
            worksheet.write(row, column, titles[index], bold)
            column = 1
            for item in data[index]:
                if item == 0:
                    pass
                else:
                    worksheet.write(row, column, item) 
                column += 1
            row += 1
        workbook.close() 
if __name__ == "__main__":
    pass
