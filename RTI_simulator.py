# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:26:57 2024

@author: brian
"""
print('##################################################################################################')
print('This is a demo for Real-Time Inference (RTI)!')
print('Welcome to the RTI-simulation program.')
print('The software is loading currently.')
print(' - Kiman B. Park, Ph.D. - ')
print('##################################################################################################')

import sys, os

import logging
import re
import winsound


import joblib

import tkinter as tk
import tkinter.ttk as ttk
from tkinter.ttk import *
from tkinter.ttk import Combobox, Progressbar, Menubutton, Style, Separator, Treeview, Label, Button, Frame
from tkinter.filedialog import asksaveasfilename, askopenfilename, askdirectory
from tkinter import scrolledtext, Menu, Canvas
from ttkthemes import ThemedTk,ThemedStyle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.figure import Figure 
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg #, NavigationToolbar2Tk

from scipy.io import loadmat
from scipy.signal import welch

##################################################################################################

'''
This is for the onefile-bundled executable file directories
'''
frozen = 'not'
if getattr(sys, 'frozen', False):
    # we are running in a bundle
    frozen = 'ever so'
    bundle_dir = sys._MEIPASS
else:
    # we are running in a normal Python environment
    try:
        bundle_dir = os.path.dirname(os.path.abspath('EdenBand'))
    except:
        bundle_dir = os.path.dirname(os.path.abspath(__file__))

path_to_data = bundle_dir + '/data_files'
#sys.path.append(path_to_data)
##################################################################################################

# This is for debugging 
logging.basicConfig(filename='tracking.log', encoding='utf-8', level=logging.DEBUG)

##################################################################################################

# a function to extract data from a single file
def get_data():
    #f = open(path_to_data + '/r_l.txt','r')
    files = [x for x in os.listdir(path_to_data) if re.search('.mat',x)]
    data = []
    for i in range(len(files)):
        file = loadmat(path_to_data + '/' + files[i])
        file['name'] = files[i]
        data.append(file)
    return data

def preprocess(data,dataset_n):
    window = 50000
    skip = 5000
    num = data[dataset_n]['NBM_TABR'][(~np.isnan(data[dataset_n]['NBM_TABR'])) & (~np.isnan(data[dataset_n]['NBM_TABR']))].shape[0]
    freq = []
    y_welch = []
    y_mean = []
    y_std = []
    for i in range(0,num-window,skip):
        freq.append(data[dataset_n]['ts_NBM_Sim'][(~np.isnan(data[dataset_n]['NBM_TABR'])) & (~np.isnan(data[dataset_n]['NBM_TABR']))][int(i+window/2)])
        test = data[dataset_n]['NBM_TABR'][(~np.isnan(data[dataset_n]['NBM_TABR'])) & (~np.isnan(data[dataset_n]['NBM_TABR']))][i:(i+window)]
        _ , S_test = welch(test, fs=250)
        y_welch.append(S_test[1])
        S_test = np.mean(test)
        y_mean.append(S_test)
        S_test = np.std(test)
        y_std.append(S_test)
        
    return freq,y_welch,y_mean,y_std

##################################################################################################


class StartUp:
    '''
    This is a loading module that executes as everything gets loaded
    '''
    def __init__(self,t):
        self.win = ThemedTk(theme=t)
        color = Style().lookup("TFrame", "background", default="black")
        self.win.configure(bg=color)
        
        self.win.title("Initializing RTI-simulation software")
        self.win.eval('tk::PlaceWindow . center')
        
        self.win.attributes('-disabled', True)
        
        self.frame = Frame(self.win)
        self.frame.grid(row=0,column=0,padx= 200,pady=100,stick="nsew")
        
        self.progStatus = tk.StringVar()
        self.progStatus.set("Initializing...")
        
        self.label_1 = Label(self.frame, textvariable=self.progStatus)
        self.label_1.grid(row=0,column=0)

        progress = Progressbar(self.frame, orient = tk.HORIZONTAL, length = 100, mode = 'determinate')
        progress.grid(row=1,column=0)

        progress['value'] = 0
        self.win.update()
        self.win.after(1000, self.progStatus.set("Loading necessary data and models"))
        progress['value'] = 20
        self.win.update()
        
        self.win.after(1000, self.progStatus.set("Loading model 1"))
        self.win.update()
        
        # Loading the model here. 
        try:
            self.model1 = joblib.load(path_to_data + '/RandomForestClassifier.pkl')
            self.win.after(1000, self.progStatus.set("model 1 successfully loaded"))
            progress['value'] = 40
            self.win.update()
        except:
            self.win.after(1000, self.progStatus.set('-'*50 + \
                                                     '\n' + \
                                                     ' '*10 + \
                                                     "failed to load the model 1" + \
                                                     '\n' + \
                                                     '-'*50))
            progress['value'] = 40
            self.model1 = ''
            self.win.update()
            
        self.win.after(1000, self.progStatus.set("Loading model 2"))
        self.win.update()
        
        # Loading the model here
        try:
            self.model2 = joblib.load(path_to_data + '/KNeighborsClassifier.pkl')
            self.win.after(1000, self.progStatus.set("model 2 successfully loaded"))
            progress['value'] = 60
            self.win.update()
        except:
            self.win.after(1000, self.progStatus.set('-'*50 + \
                                                     '\n' + \
                                                     ' '*10 + \
                                                     "failed to load the model 2" + \
                                                     '\n' + \
                                                     '-'*50))
            progress['value'] = 60
            self.model2 = ''
            self.win.update()
            
        self.win.after(1000, self.progStatus.set("Loading the data"))
        self.win.update()
        
        # Loading the data here
        try:
            self.data = get_data()
            self.win.after(1000, self.progStatus.set("Data successfully loaded"))
            progress['value'] = 80
            self.win.after(1000, self.progStatus.set("Loading the data"))
            self.win.update()
            
        except:
            self.win.after(1000, self.progStatus.set('-'*50 + \
                                                     '\n' + \
                                                     ' '*10 + \
                                                     "failed to load the data" + \
                                                     '\n' + \
                                                     '-'*50))
            progress['value'] = 80
            self.win.update()
            
        
        progress['value'] = 90
        self.win.update()
        
        self.win.after(1000, self.progStatus.set("Ready"))
        
        
        progress['value'] = 100
        self.win.update()
        self.win.after(500, self.win.destroy)
        
    
    
##################################################################################################


def listStyle():
    style_list = ThemedStyle().theme_names()
    return style_list

def StyleChanged(self,win,style_list):

    # Get the text of the item whose Id is stored in `my_iid`.
    selection = self.listbox.selection()
    chosen_style = style_list[int(selection[0])]
    win.set_theme(chosen_style)
    color = Style().lookup("TFrame", "background", default="black")
    win.configure(bg=color)
    self.styling.destroy()


def ChangeStyle(self,win):
    self.styling = tk.Tk()
    self.styling.wm_title("Style")
    
    self.style_frame = Frame(self.styling)
    self.style_frame.grid(row=0,column=0)
    
    self.scrollbar = Scrollbar(self.style_frame)
    self.scrollbar.grid(row=0, column=1, sticky='ns')
    
    self.listbox = Treeview(self.style_frame,
                       show="tree")
    self.listbox.grid(row=0,column=0)
    
    self.listbox.config(yscrollcommand=self.scrollbar.set)
    self.scrollbar.config(command=self.listbox.yview)
    
    style_list = listStyle()
    for i,k in enumerate(style_list):
        self.listbox.insert('', 'end', text=k, iid=i)
    
    self.style_button = Button(self.style_frame, text="Change style", command=lambda: StyleChanged(self,win,style_list))
    self.style_button.grid(row=0,column=2)
    


def empty():
    return None

def ModelParam1():
    param = joblib.load(path_to_data + '/RandomForestClassifier_best_param.pkl')
    model = joblib.load(path_to_data + '/RandomForestClassifier.pkl')
    param1 = tk.Tk()
    param1.wm_title("Random Forest Classifier")
    param1_text = scrolledtext.ScrolledText(param1,width=80,height=15)
    param1_text.grid(row=0,column=0,pady=5)
    param1_text.insert(tk.END,'The model:')
    param1_text.insert(tk.END,'\n')
    param1_text.insert(tk.END,model)
    param1_text.insert(tk.END,'\n')
    param1_text.insert(tk.END,'Best parameters:')
    param1_text.insert(tk.END,'\n')
    param1_text.insert(tk.END,param)
        
def ModelParam2():
    model = joblib.load(path_to_data + '/KNeighborsClassifier.pkl')
    param = joblib.load(path_to_data + '/KNeighborsClassifier_best_param.pkl')
    param2 = tk.Tk()
    param2.wm_title("K-Nearest Neighbor Classifier")
    param2_text = scrolledtext.ScrolledText(param2,width=80,height=15)
    param2_text.grid(row=0,column=0,pady=5)
    param2_text.insert(tk.END,'The model:')
    param2_text.insert(tk.END,'\n')
    param2_text.insert(tk.END,model)
    param2_text.insert(tk.END,'\n')
    param2_text.insert(tk.END,'Best parameters:')
    param2_text.insert(tk.END,'\n')
    param2_text.insert(tk.END,param)

def OpenLog():
    f = open('tracking.log','r')
    lines = f.readlines()
    f.close()
    log = tk.Tk()
    log.wm_title("Logs")
    logging = scrolledtext.ScrolledText(log,width=80,height=15)
    logging.grid(row=0,column=0,pady=5)
    for line in lines:
        logging.insert(tk.END,line)
        
def ClearLog():
    f = open('tracking.log','w+')
    f.write('')
    f.close()
            
def Clearplot(self):
    
    self.frame2.destroy()
    
    self.frame2 = Frame(self.frame)
    self.frame2.grid(row=1,column=0,padx=5)
     
    self.fig = Figure(figsize=(14,6.5))
    self.ax1 = self.fig.add_subplot(411)
    self.ax2 = self.ax1.twinx()
    self.ax3 = self.fig.add_subplot(412)
    self.ax4 = self.ax3.twinx()
    self.ax5 = self.fig.add_subplot(413)
    self.ax6 = self.ax5.twinx()
    self.ax7 = self.fig.add_subplot(414)
    self.ax8 = self.ax7.twinx()

    self.output = FigureCanvasTkAgg(self.fig, master=self.frame2)
    self.output.draw()
    self.output.get_tk_widget().pack()

def Replot(self):
    
    self.frame2.destroy()
    
    self.frame2 = Frame(self.frame)
    self.frame2.grid(row=1,column=0,padx=5)
    
    self.fig = Figure(figsize=(14,6.5))
    self.ax1 = self.fig.add_subplot(411)
    self.ax2 = self.ax1.twinx()
    self.ax3 = self.fig.add_subplot(412)
    self.ax4 = self.ax3.twinx()
    self.ax5 = self.fig.add_subplot(413)
    self.ax6 = self.ax5.twinx()
    self.ax7 = self.fig.add_subplot(414)
    self.ax8 = self.ax7.twinx()
            
    
def Plot(self,
         data,
         ):
    
    Replot(self)
    try:
        subj = int(self.dataset_n.get().split('#')[1])
        freq, y_welch, y_mean, y_std = preprocess(data,subj)
        
        norm = np.mean(data[subj]['NBM_TABR'][(~np.isnan(data[subj]['NBM_TABR'])) & (~np.isnan(data[subj]['NBM_TABR']))])
        self.ax1.plot(data[subj]['ts_NBM_Sim'],data[subj]['NBM_TABR'],alpha=0.8,c='orange')
        self.ax1.axhline(norm,alpha=0.2,linestyle='--',c='black')
        self.ax1.set_title('Raw NBM')
        self.ax1.set_ylabel('Fatigue')
        self.ax1.set_xlabel('Seconds')
        self.ax2.scatter(data[subj]['KSS'][1:,0],data[subj]['KSS'][1:,1],alpha=0.8,c='black')
        self.ax2.set_ylim(0,10)
        for i in range(len(data[subj]['KSS'][1:,0])):
            self.ax2.axvline(data[subj]['KSS'][(1+i),0],ymax=data[subj]['KSS'][(1+i),1]/10,alpha=0.3,linestyle='--',c='r')
        self.ax2.set_ylabel('KSS scale')
        
        norm = np.mean(y_mean)
        self.ax3.plot(freq[:-20],np.array(y_mean[:-20]),alpha=0.8,c='orange')
        self.ax3.axhline(norm,alpha=0.2,linestyle='--',c='black')
        self.ax3.set_title('Mean Raw NBM')
        self.ax3.set_ylabel('NBM Signal Mean')
        self.ax3.set_xlabel('Seconds')
        self.ax4.scatter(data[subj]['KSS'][1:,0],data[subj]['KSS'][1:,1],alpha=0.8,c='black')
        self.ax4.set_ylim(0,10)
        for i in range(len(data[subj]['KSS'][1:,0])):
            self.ax4.axvline(data[subj]['KSS'][(1+i),0],ymax=data[subj]['KSS'][(1+i),1]/10,alpha=0.3,linestyle='--',c='r')
        self.ax4.set_ylabel('KSS scale')
        
        norm = np.mean(y_std)
        self.ax5.plot(freq[:-20],y_std[:-20],alpha=0.8,c='orange')
        self.ax5.axhline(norm,alpha=0.2,linestyle='--',c='black')
        self.ax5.set_title('Raw NBM Standard Deviation')
        self.ax5.set_ylabel('Standard deviation\nof NBM signal')
        self.ax5.set_xlabel('Seconds')
        self.ax6.scatter(data[subj]['KSS'][1:,0],data[subj]['KSS'][1:,1],alpha=0.8,c='black')
        self.ax6.set_ylim(0,10)
        for i in range(len(data[subj]['KSS'][1:,0])):
            self.ax6.axvline(data[subj]['KSS'][(1+i),0],ymax=data[subj]['KSS'][(1+i),1]/10,alpha=0.3,linestyle='--',c='r')
        self.ax6.set_ylabel('KSS scale')
        
        norm = np.mean(y_welch)
        self.ax7.plot(freq[:-20],y_welch[:-20],alpha=0.8,c='orange')
        self.ax7.axhline(norm,alpha=0.2,linestyle='--',c='black')
        self.ax7.set_title('Power Spectral Density')
        self.ax7.set_ylabel('Power Spectral Density')
        self.ax7.set_xlabel('Seconds')
        self.ax8.scatter(data[subj]['KSS'][1:,0],data[subj]['KSS'][1:,1],alpha=0.8,c='black')
        self.ax8.set_ylim(0,10)
        for i in range(len(data[subj]['KSS'][1:,0])):
            self.ax8.axvline(data[subj]['KSS'][(1+i),0],ymax=data[subj]['KSS'][(1+i),1]/10,alpha=0.3,linestyle='--',c='r')
        self.ax8.set_ylabel('KSS scale')
        
        self.fig.tight_layout()
        
        self.output = FigureCanvasTkAgg(self.fig, master=self.frame2)
        self.output.draw()
        self.output.get_tk_widget().pack()
    except:
        Clearplot(self)
        pass


def stopping_sim(sim_state):
    sim_state.set(False)

def simulate(self,
             win,
             data,
             model1,
             model2,
             ):
    
    subj = int(self.dataset_n.get().split('#')[1])
    t = win.current_theme
    
    popup_sim = ThemedTk(theme=t)
    color_sim = Style().lookup("TFrame", "background", default="black")
    popup_sim.configure(bg=color_sim)
    popup_sim.wm_title("Simulation")
    
    #############################################################################################
    
    frame_sim = Frame(popup_sim)
    frame_sim.grid(row=0,column=0,padx=50)
    
    frame_sim1 = Frame(popup_sim)
    frame_sim1.grid(row=1,column=0)
    
    #############################################################################################
    
    time_label = Label(frame_sim,text='Time (s): ')   
    time_label.grid(row=0,column=0)
    
    time_sim = Label(frame_sim,text='')
    time_sim.grid(row=0,column=1,padx=5)
    
    sig_label = Label(frame_sim,text='Fatigue signal: ')   
    sig_label.grid(row=1,column=0)
    
    sig_sim = Label(frame_sim,text='')
    sig_sim.grid(row=1,column=1,padx=5,pady=2)
    
    KSS_label = Label(frame_sim,text='KSS scale (1-10): ')
    KSS_label.grid(row=2,column=0,pady=2)
    
    KSS_sim = Label(frame_sim,text='')
    KSS_sim.grid(row=2,column=1,padx=5,pady=2)
    
    sim_state = tk.BooleanVar()
    sim_state.set(True)

    stop_sim = Button(frame_sim,text='Stop the simulation',command=lambda: stopping_sim(sim_state))
    stop_sim.grid(row=3,column=0,rowspan=2)
    
    #############################################################################################
    
    self.fig_sim = plt.figure(figsize=(15,3))
    self.ax_sim = self.fig_sim.add_subplot(111)
    self.ax_sim2 = self.ax_sim.twinx()
    
    self.ax_sim.plot(data[subj]['ts_NBM_Sim'],data[subj]['NBM_TABR'],alpha=0.8,c='orange')
    self.ax_sim.set_title('Raw NBM')
    self.ax_sim.set_ylabel('Fatigue')
    self.ax_sim.set_xlabel('Seconds')
    self.ax_sim2.scatter(data[subj]['KSS'][1:,0],data[subj]['KSS'][1:,1],alpha=0.8,c='black')
    self.ax_sim2.set_ylim(0,10)
    for i in range(len(data[subj]['KSS'][1:,0])):
        self.ax_sim2.axvline(data[subj]['KSS'][(1+i),0],ymax=data[subj]['KSS'][(1+i),1]/10,alpha=0.3,linestyle='--',c='r')
    self.ax_sim2.set_ylabel('KSS scale')
    
    self.fig_sim.tight_layout()
    
    #############################################################################################
    
    canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=frame_sim1)
    canvas_sim.draw()
    canvas_sim.get_tk_widget().pack()
    
    threshold_label = Label(frame_sim,text='Set Threshold (1-10 in KSS scale):')
    threshold_label.grid(row=0,column=2,padx=5,pady=2)
    
    thres = tk.StringVar()
    thres_set = Combobox(frame_sim, textvariable=thres)
    thres_set['values'] = [str(x) for x in range(1,11)]
    thres_set.current(6)
    thres_set.grid(row=1,column=2,padx=5,pady=2)
    
    alert_label = Label(frame_sim,text='Waiting for KSS measurement...')
    alert_label.grid(row=3,column=2,padx=10,pady=2)
    alert_label.config(width=30)
    alert_label.config(font=("Courier", 15))
    
    #############################################################################################
    
    sim_x = np.array([])
    sig = np.array([])
    KSS = np.array([])
    time=0
    alert_line = self.ax_sim2.axhline(int(thres_set.get()),alpha=0.6,linestyle='--',c='red')
    measure_window = self.ax_sim.axvspan(0,0,alpha=0.1,facecolor='b')
    alert_window = self.ax_sim2.axhspan(0,0,alpha=0.1,facecolor='r')
    try:
        while sim_state.get():
            for i in range(int(data[subj]['ts_NBM_Sim'].ravel()[0]),len(data[subj]['NBM_TABR'].ravel()),1000):
                time  = data[subj]['ts_NBM_Sim'].ravel()[i]
                sig = np.append(sig,data[subj]['NBM_TABR'].ravel()[i])
                sim_x = np.append(sim_x,time).ravel()
                measure_window.set_xy([[time-75,0],[time-75,10],[time,10],[time,0],[time-75,0]])
                alert_window.set_xy([[-500,int(thres_set.get())],[-500,10],[7000,10],[7000,int(thres_set.get())],[-1000,int(thres_set.get())]])
                alert_line.set_ydata([int(thres_set.get())])
                try:
                    pred = model1.predict(np.array(data[subj]['NBM_TABR'].ravel()[(i-18626):i]).reshape(1,18626))
                except:
                    pred = 0
                    alert_label.config(text='Waiting for KSS measurement...')
                    alert_label.config(foreground="black")
                    alert_label.config(font=("Courier", 15))
                if pred >= int(thres_set.get()):
                    alert_label.config(text='Alert: You are fatigued!')
                    alert_label.config(foreground="red")
                    alert_label.config(font=("Courier", 23))
                    winsound.Beep(700, 1000)
                else:
                    alert_label.config(text='You are awake and safe')
                    alert_label.config(foreground='green')
                    alert_label.config(font=("Courier", 15))
                
                KSS = np.append(KSS,pred).ravel()
                time_sim.config(text=sim_x[-1])
                sig_sim.config(text=sig[-1])
                KSS_sim.config(text=KSS[-1])
                if not sim_state.get():
                    break
                else:
                    self.ax_sim2.scatter(sim_x,KSS,alpha=0.2,c='g',marker='*')
                    
                canvas_sim.draw() # draw
                canvas_sim.flush_events() # deal with resize
                
            del sim_x,sig,KSS,
            popup_sim.destroy()
    except KeyboardInterrupt:
        pass
    
    #toolbar = NavigationToolbar2Tk(canvas_sim, popup_sim,)
    #toolbar.update()
    #canvas_sim._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    
    
def Close(self,win): 
    win.destroy() 
    win.quit()
    
    
##################################################################################################
class MyWindow():
    def __init__(self, 
                 win,
                 model1,
                 model2,                
                 data,
                 ):
        
        win.after(500)
        
        ##################################################################################################
        
        self.menubar = Menu(win)
        # create a pulldown menu, and add it to the menu bar
        self.filemenu = Menu(self.menubar, tearoff=0)
        
        self.filemenu.config(bg=color,fg='black',activebackground='black',activeforeground='white',relief=tk.FLAT)
        
        self.filemenu.add_command(label="Change the style", command=lambda: ChangeStyle(self,win))
        self.menubar.add_cascade(label="Work Space", menu=self.filemenu)
        
        self.paramMenu = Menu(self.menubar,tearoff=0)
        self.paramMenu.add_command(label="Open to see 1st model parameters", command=ModelParam1)
        self.paramMenu.add_command(label="Open to see 2nd model parameters", command=ModelParam2)
        self.menubar.add_cascade(label="Parameters", menu=self.paramMenu)
        
        # create more pulldown menus ()
        self.editmenu = Menu(self.menubar, tearoff=0)
        self.editmenu.add_command(label="Open the Log", command=OpenLog)
        self.editmenu.add_command(label="Clear the log", command=ClearLog)
        self.menubar.add_cascade(label="Diagnostics", menu=self.editmenu)

        self.helpmenu = Menu(self.menubar, tearoff=0)
        self.helpmenu.add_command(label="Operation Manual", command=empty)
        self.menubar.add_cascade(label="Help", menu=self.helpmenu)
        
        # display the menu
        win.config(menu=self.menubar)
        
        ####################################################################################
        
        self.frame = Frame(win)
        self.frame.grid(row=0,column=0)
                
        self.frame1 = Frame(self.frame)
        self.frame1.grid(row=0,column=0,padx=500)
        
        self.frame2 = Frame(self.frame)
        self.frame2.grid(row=1,column=0,padx=5)
        ####################################################################################
        
        self.dataset = Label(self.frame1, text = 'Data set number: ')
        self.dataset.grid(row=1,column=0,pady=5)
        
        n = tk.StringVar()
        self.dataset_n = Combobox(self.frame1, textvariable=n)
        self.dataset_n['values'] = ['subject #' + str(x) for x in range(len(data))]
        self.dataset_n.grid(row=1,column=1,pady=5)
        
        self.fig = Figure(figsize=(14,6.5))
        self.ax1 = self.fig.add_subplot(411)
        self.ax2 = self.ax1.twinx()
        self.ax3 = self.fig.add_subplot(412)
        self.ax4 = self.ax3.twinx()
        self.ax5 = self.fig.add_subplot(413)
        self.ax6 = self.ax5.twinx()
        self.ax7 = self.fig.add_subplot(414)
        self.ax8 = self.ax7.twinx()
        
        self.ax1.set_title('Raw NBM')
        self.ax1.set_ylabel('Fatigue')
        self.ax1.set_xlabel('Seconds')
        self.ax2.set_ylim(0,10)

        
        self.ax3.set_title('Mean Raw NBM')
        self.ax3.set_ylabel('NBM Signal Mean')
        self.ax3.set_xlabel('Seconds')
        self.ax4.set_ylim(0,10)
        self.ax4.set_ylabel('KSS scale')
        
        self.ax5.set_title('Raw NBM Standard Deviation')
        self.ax5.set_ylabel('Standard deviation\nof NBM signal')
        self.ax5.set_xlabel('Seconds')
        self.ax6.set_ylim(0,10)
        self.ax6.set_ylabel('KSS scale')
        
        self.ax7.set_title('Power Spectral Density')
        self.ax7.set_ylabel('Power Spectral Density')
        self.ax7.set_xlabel('Seconds')
        self.ax8.set_ylim(0,10)
        self.ax8.set_ylabel('KSS scale')
        
        self.fig.tight_layout()
        
        self.output = FigureCanvasTkAgg(self.fig, master=self.frame2)
        self.output.draw()
        self.output.get_tk_widget().pack()
        
        #self.fig_scrollbar = Scrollbar(self.frame2, command=self.output.yview)
        #self.fig_scrollbar.grid(row=0,column=1)
        
        #self.output.config(yscrollcommand = self.fig_scrollbar.set)
        
        self.plotting = Button(self.frame1,text='Plot',command=lambda: Plot(self,
                                                                            data,
                                                                            ))
        self.plotting.grid(row=2,column=1,pady=5)
        
        self.clearplotting = Button(self.frame1,text='Clear Plot',command=lambda: Clearplot(self,
                                                                                            ))
        self.clearplotting.grid(row=2,column=2,pady=5)
        
        self.simulateplotting = Button(self.frame1,text='Simulation',command=lambda: simulate(self,
                                                                                           win,
                                                                                           data,
                                                                                           model1,
                                                                                           model2
                                                                                           ))
        #self.insert_lig.place(x=10,y=180)
        self.simulateplotting.grid(row=2,column=3,pady=5)
        
        self.exit_button = Button(self.frame1, text="Exit", command=lambda: Close(self,win)) 
        self.exit_button.grid(row=0,column=4) 
        
        ####################################################################################
        
        
        
    

        

##################################################################################################
if __name__ == '__main__':
    '''
    This is the main program executing everything
    '''
    # Various styles we can use
    '''
    styles = ThemedStyle().tk.call('ttk::themes')
    
    t = styles[12] # 'arc'
    t = styles[17] # 'yaru'
    t = styles[14] # 'radiance'
    t = styles[9] # 'equilux'
    '''
    t = 'arc'
    start = StartUp(t)
    
    #window = tk.Tk()
    
    #style = ThemedStyle(window)
    print('Opening the main window')
    window = ThemedTk(theme=t)
    #window.geometry("1200x600+0-0")
    color = Style().lookup("TFrame", "background", default="black")
    window.configure(bg=color)
    window.state('zoomed')
    
    #style.theme_use(t)
    #color = Style().lookup("TFrame", "background", default="black")
    #window.configure(bg=color)
        
    try:
        mywin=MyWindow(window,
                       start.model1,
                       start.model2,
                       start.data,
                       )
        window.title('RTI-simulator')
    
        window.eval('tk::PlaceWindow . center')   
        
        window.quit()
        
        window.mainloop()
    except:
        pass
    
    
    
    
    
    
    
    
    
