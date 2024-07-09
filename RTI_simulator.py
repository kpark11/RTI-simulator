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
import mne

from pyriemann.embedding import SpectralEmbedding
from pyriemann.estimation import Covariances

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
    bundle_dir = os.path.dirname(os.path.abspath(__file__))
path_to_data = bundle_dir + '/data_files'
#sys.path.append(path_to_data)
##################################################################################################

# This is for debugging 
logging.basicConfig(filename='tracking.log', encoding='utf-8', level=logging.DEBUG)

##################################################################################################


chnames = [
        'Fp1',
        'Fp2',
        'Fc5',
        'Fz',
        'Fc6',
        'T7',
        'Cz',
        'T8',
        'P7',
        'P3',
        'Pz',
        'P4',
        'P8',
        'O1',
        'Oz',
        'O2',
        'stim']

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

def processData(data):
    S = data['SIGNAL'][:, 1:17]
    stim_close = data['SIGNAL'][:, 17]
    stim_open = data['SIGNAL'][:, 18]
    stim = 1 * stim_close + 2 * stim_open
    chtypes = ['eeg'] * 16 + ['stim']
    X = np.concatenate([S, stim[:, None]], axis=1).T

    info = mne.create_info(ch_names=chnames, sfreq=512,
                           ch_types=chtypes,
                           verbose=False)
    raw = mne.io.RawArray(data=X, info=info, verbose=False)
    # filter data and resample
    fmin = 3
    fmax = 40
    raw.filter(fmin, fmax, verbose=False)
    raw.resample(sfreq=128, verbose=False)

    # detect the events and cut the signal into epochs
    events = mne.find_events(raw=raw, shortest_event=1, verbose=False)
    event_id = {'closed': 1, 'open': 2}
    epochs = mne.Epochs(raw, events, event_id, tmin=-0.2, tmax=0.5, baseline=None,
                        verbose=False,preload=True)
    epochs.pick_types(eeg=True,verbose=False)
    return raw,epochs,events

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
        
        self.win.after(1000, self.progStatus.set("Loading the RF model"))
        self.win.update()
        
        # Loading the model here. 
        try:
            print('loading the RF model')
            self.model1 = joblib.load(path_to_data + '/RF_model.pkl')
            print('RF model successfully loaded')
            self.win.after(1000, self.progStatus.set("RF model successfully loaded"))
            progress['value'] = 40
            self.win.update()
        except:
            print('RF model failed to load')
            self.win.after(1000, self.progStatus.set('-'*50 + \
                                                     '\n' + \
                                                     ' '*10 + \
                                                     "failed to load the RF model" + \
                                                     '\n' + \
                                                     '-'*50))
            progress['value'] = 40
            self.model1 = ''
            self.win.update()
            
        self.win.after(1000, self.progStatus.set("Loading the SVC model"))
        self.win.update()
        
        # Loading the model here
        try:
            print('loading the SVC model')
            self.model2 = joblib.load(path_to_data + '/SVC_model.pkl')
            print('SVC model successfully loaded')
            self.win.after(1000, self.progStatus.set("SVC model successfully loaded"))
            progress['value'] = 60
            self.win.update()
        except:
            print('SVC model failed to load')
            self.win.after(1000, self.progStatus.set('-'*50 + \
                                                     '\n' + \
                                                     ' '*10 + \
                                                     "failed to load the SVC model" + \
                                                     '\n' + \
                                                     '-'*50))
            progress['value'] = 60
            self.model2 = ''
            self.win.update()
            
        self.win.after(1000, self.progStatus.set("Loading the LSTM model"))
        self.win.update()
        
        
        try:
            print('loading LSTM model')
            self.model3 = joblib.load(path_to_data + '/LSTM_model.pkl')
            print('SVC model successfully loaded')
            self.win.after(1000, self.progStatus.set("LSTM model successfully loaded"))
            progress['value'] = 80
            self.win.update()
        except:
            print('LSTM model failed to load')
            self.win.after(1000, self.progStatus.set('-'*50 + \
                                                     '\n' + \
                                                     ' '*10 + \
                                                     "failed to load the LSTM model" + \
                                                     '\n' + \
                                                     '-'*50))
            progress['value'] = 80
            self.model2 = ''
            self.win.update()
            
        self.win.after(1000, self.progStatus.set("Loading the data"))
        self.win.update()
        
        # Loading the data here
        try:
            print('loading data')
            self.data = get_data()
            print('Data successfully loaded')
            self.win.after(1000, self.progStatus.set("Data successfully loaded"))
            progress['value'] = 90
            self.win.update()
            
        except:
            print('data failed to load')
            self.win.after(1000, self.progStatus.set('-'*50 + \
                                                     '\n' + \
                                                     ' '*10 + \
                                                     "failed to load the data" + \
                                                     '\n' + \
                                                     '-'*50))
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
    model = joblib.load(path_to_data + '/RF_model.pkl')
    param1 = tk.Tk()
    param1.wm_title("RF model")
    param1_text = scrolledtext.ScrolledText(param1,width=80,height=15)
    param1_text.grid(row=0,column=0,pady=5)
    param1_text.insert(tk.END,'The model:')
    param1_text.insert(tk.END,'\n')
    param1_text.insert(tk.END,model)
    param1_text.insert(tk.END,'\n')
        
def ModelParam2():
    model = joblib.load(path_to_data + '/SVC_model.pkl')
    param2 = tk.Tk()
    param2.wm_title("SVC model")
    param2_text = scrolledtext.ScrolledText(param2,width=80,height=15)
    param2_text.grid(row=0,column=0,pady=5)
    param2_text.insert(tk.END,'The model:')
    param2_text.insert(tk.END,'\n')
    param2_text.insert(tk.END,model)
    param2_text.insert(tk.END,'\n')

def ModelParam3():
    model = joblib.load(path_to_data + '/LSTM_model.pkl')
    param2 = tk.Tk()
    param2.wm_title("LSTM model")
    param2_text = scrolledtext.ScrolledText(param2,width=80,height=15)
    param2_text.grid(row=0,column=0,pady=5)
    param2_text.insert(tk.END,'The model:')
    param2_text.insert(tk.END,'\n')
    param2_text.insert(tk.END,model)
    param2_text.insert(tk.END,'\n')

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
    
    self.ax = self.fig.subplot_mosaic([['TopLeft', 'Right'],['BottomLeft','Right']],
                              gridspec_kw={'width_ratios':[10, 5]})
    self.ax2 = self.ax['TopLeft'].twinx()
    
    self.ax['TopLeft'].set_title('Raw Data')
    self.ax['TopLeft'].set_ylabel('Signal (Arb. Unit)')
    self.ax['TopLeft'].set_xlabel('Seconds')
    self.ax2.set_ylim(0,3)
    self.ax2.set_ylabel('events (1 or 2)')

    self.ax['BottomLeft'].set_title('Power Spectral Density')
    self.ax['BottomLeft'].set_ylabel('Signal (Arb. Unit)')
    self.ax['BottomLeft'].set_xlabel('Frequency (Hz)')
    
    self.ax['Right'].set_title('Spectral embedding with covariances')
    self.ax['Right'].set_ylabel('')
    self.ax['Right'].set_xlabel('')
    
    self.fig.tight_layout()

    self.output = FigureCanvasTkAgg(self.fig, master=self.frame2)
    self.output.draw()
    self.output.get_tk_widget().pack()

def Replot(self):
    
    self.frame2.destroy()
    
    self.frame2 = Frame(self.frame)
    self.frame2.grid(row=1,column=0,padx=5)
    
    self.fig = Figure(figsize=(14,6.5))
    
    self.ax = self.fig.subplot_mosaic([['TopLeft', 'Right'],['BottomLeft','Right']],
                              gridspec_kw={'width_ratios':[10, 5]})
    self.ax2 = self.ax['TopLeft'].twinx()
    
    self.ax['TopLeft'].set_title('Raw Data')
    self.ax['TopLeft'].set_ylabel('Signal (Arb. Unit)')
    self.ax['TopLeft'].set_xlabel('Seconds')
    self.ax2.set_ylim(0,3)
    self.ax2.set_ylabel('events (1 or 2)')

    self.ax['BottomLeft'].set_title('Power Spectral Density')
    self.ax['BottomLeft'].set_ylabel('Signal (Arb. Unit)')
    self.ax['BottomLeft'].set_xlabel('Frequency (Hz)')
    
    self.ax['Right'].set_title('Spectral embedding with covariances')
    self.ax['Right'].set_ylabel('')
    self.ax['Right'].set_xlabel('')

    
def Plot(self,
         data,
         ):
    
    Replot(self)
    try:
        subj = int(self.dataset_n.get().split('#')[1])
        ch = str(self.ch_n.get().split(' ')[1])
        raw,epochs,events = processData(data[subj])
        epochs.load_data().pick(ch)
        
        norm = np.mean(raw.get_data()[chnames.index(ch)])
        self.ax['TopLeft'].plot(raw.get_data()[chnames.index(ch)],alpha=0.8,c='orange')
        self.ax['TopLeft'].axhline(norm,alpha=0.2,linestyle='--',c='black')
        self.ax2.scatter(events[:,0],events[:,-1],alpha=0.5,c='black')
        self.ax2.set_ylim(0,3)
        for i in range(len(events[:,-1])):
            self.ax2.axvline(events[i,0],ymin=0,ymax=events[i,-1],alpha=0.3,linestyle='--',c='r')
        self.ax2.set_ylabel('Events (1 or 2)')
        
        # estimate the averaged spectra for each condition
        X_closed = epochs['closed'].get_data(verbose=False)
        f, S_closed = welch(X_closed, fs=epochs.info['sfreq'], axis=2)
        S_closed = np.mean(S_closed, axis=0).squeeze()
        X_opened = epochs['open'].get_data(verbose=False)
        f, S_opened = welch(X_opened, fs=epochs.info['sfreq'], axis=2)
        S_opened = np.mean(S_opened, axis=0).squeeze()  
        
        self.ax['BottomLeft'].plot(f,S_opened,alpha=0.8,c='red',label='opened')
        self.ax['BottomLeft'].plot(f,S_closed,alpha=0.8,c='blue',label='closed')
        self.ax['BottomLeft'].legend()
        
        X = epochs.get_data(verbose=False)
        y = events[:,-1]
        
        C = Covariances(estimator='lwf').fit_transform(X)
        emb = SpectralEmbedding(metric='riemann').fit_transform(C)
        colors = {1: 'r', 2: 'b'}
        for embi, yi in zip(emb, y):
            self.ax['Right'].scatter(embi[0], embi[1], s=120, c=colors[yi])
        labels = {2: 'open', 1: 'closed'}
        for yi in np.unique(y):
            self.ax['Right'].scatter([], [], c=colors[yi], label=labels[yi])
        self.ax['Right'].legend()
        
        self.fig.tight_layout()
        
        self.output = FigureCanvasTkAgg(self.fig, master=self.frame2)
        self.output.draw()
        self.output.get_tk_widget().pack()
    except:
        Clearplot(self)
        pass

def choose_model(start_sim,stop_sim):
    start_sim.configure(state='normal')
    stop_sim.configure(state='normal')
    
    
def stopping_sim(sim_state):
    sim_state.set(False)

def starting_sim(sim_state,
                 self,
                 model1,
                 model2,
                 model3,
                 popup_sim,
                 subj,
                 ch,
                 raw,
                 epochs,
                 events,
                 incoming_sig,
                 alert_label,
                 thres_set,
                 time_sim,
                 sig_sim,
                 KSS_sim,
                 canvas_sim,
                 select_model_set):
    sim_state.set(True)
    sim_x = np.array([])
    sig = np.array([])
    KSS = np.array([])
    time=0
    thres_num = int(thres_set.get())
    alert_line = self.ax_sim2.axhline(thres_set.get(),alpha=0.6,linestyle='--',c='red')
    measure_window = self.ax_sim.axvspan(0,0,alpha=0.1,facecolor='b')
    alert_window = self.ax_sim2.axhspan(0,0,alpha=0.1,facecolor='r')
    try:
        while sim_state.get():
            for i in range(len(incoming_sig)):
                time  = i
                sig = np.append(sig,incoming_sig[i])
                sim_x = np.append(sim_x,time).ravel()
                measure_window.set_xy([[time-91,0],[time-91,10],[time,10],[time,0],[time-91,0]])
                alert_window.set_xy([[-500,thres_num],[-500,10],[7000,10],[7000,thres_num],[-1000,thres_num]])
                alert_line.set_ydata([thres_num])
                try:
                    if select_model_set.get() == 1:
                        pred = model1.predict()
                    elif select_model_set.get() == 2:
                        pred = model2.predict()
                    else:
                        pred = model3.predict()
                except:
                    pred = 0
                    alert_label.config(text='Waiting for KSS measurement...')
                    alert_label.config(foreground="black")
                    alert_label.config(font=("Courier", 15))
                if pred >= thres_num:
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
    

def simulate(self,
             win,
             data,
             model1,
             model2,
             model3
             ):
    
    subj = int(self.dataset_n.get().split('#')[1])
    ch = str(self.ch_n.get().split(' ')[1])
    raw,epochs,events = processData(data[subj])
    incoming_sig = raw.get_data()[chnames.index(ch)]
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
    
    sig_label = Label(frame_sim,text='EEG signal: ')   
    sig_label.grid(row=1,column=0)
    
    sig_sim = Label(frame_sim,text='')
    sig_sim.grid(row=1,column=1,padx=5,pady=2)
    
    KSS_label = Label(frame_sim,text='KSS scale (1-10): ')
    KSS_label.grid(row=2,column=0,pady=2)
    
    KSS_sim = Label(frame_sim,text='')
    KSS_sim.grid(row=2,column=1,padx=5,pady=2)
    
    sim_state = tk.BooleanVar()
    sim_state.set(False)
    
    #############################################################################################
    
    self.fig_sim = Figure(figsize=(14,6.5))
    self.ax_sim = self.fig.add_subplot(111)
    self.ax_sim2 = self.ax_sim.twinx()
    
    self.ax_sim.set_title('Raw Data')
    self.ax_sim.set_ylabel('Signal (Arb. Unit)')
    self.ax_sim.set_xlabel('Seconds')
    self.ax_sim.set_ylim(0,2)
    
    norm = np.mean(raw.get_data()[chnames.index(ch)])
    self.ax_sim.plot(incoming_sig,alpha=0.8,c='orange')
    self.ax_sim.axhline(norm,alpha=0.2,linestyle='--',c='black')
    self.ax_sim2.scatter(events[:,0],events[:,-1],alpha=0.5,c='black')
    self.ax_sim2.set_ylim(0,3)
    for i in range(len(events[:,-1])):
        self.ax2.axvline(events[i,0],ymin=0,ymax=events[i,-1],alpha=0.3,linestyle='--',c='r')
    
    self.fig_sim.tight_layout()
    
    canvas_sim = FigureCanvasTkAgg(self.fig_sim, master=frame_sim1)
    canvas_sim.draw()
    canvas_sim.get_tk_widget().pack()
    
    #############################################################################################
    
    select_model_label = Label(frame_sim,text='Pick a model')
    select_model_label.grid(row=0,column=2,padx=5,pady=2)
    
    select_model = tk.StringVar()
    select_model_set = Combobox(frame_sim, textvariable=select_model)
    select_model_set['values'] = [str(x) for x in range(1,4)]
    select_model_set.current(1)
    select_model_set.grid(row=1,column=2,padx=5,pady=2)
    select_model_set.bind("<<ComboboxSelected>>", lambda event: choose_model(start_sim,stop_sim))
    
    threshold_label = Label(frame_sim,text='Set Threshold (1-10 in KSS scale):')
    threshold_label.grid(row=0,column=3,padx=5,pady=2)
    
    thres = tk.StringVar()
    thres_set = Combobox(frame_sim, textvariable=thres)
    thres_set['values'] = [str(x) for x in range(1,11)]
    thres_set.current(6)
    thres_set.grid(row=1,column=3,padx=5,pady=2)
    
    alert_label = Label(frame_sim,text='Waiting for EEG measurement...')
    alert_label.grid(row=3,column=2,padx=10,pady=2)
    alert_label.config(width=30)
    alert_label.config(font=("Courier", 15))
    
    #############################################################################################
    
    start_sim = Button(frame_sim,text='Start the simulation', state='disable', command=lambda: starting_sim(sim_state,
                                                                                          self,
                                                                                          model1,
                                                                                          model2,
                                                                                          model3,
                                                                                          popup_sim,
                                                                                          subj,
                                                                                          ch,
                                                                                          raw,
                                                                                          epochs,
                                                                                          events,
                                                                                          incoming_sig,
                                                                                          alert_label,
                                                                                          thres_set,
                                                                                          time_sim,
                                                                                          sig_sim,
                                                                                          KSS_sim,
                                                                                          canvas_sim,
                                                                                          select_model_set))
    start_sim.grid(row=3,column=0,rowspan=2)

    stop_sim = Button(frame_sim,text='Stop the simulation',  state='disable', command=lambda: stopping_sim(sim_state))
    stop_sim.grid(row=3,column=1,rowspan=2)
    
    
def Close(self,win): 
    win.destroy() 
    win.quit()
    

def ch_changed(self):
    print("Channel updated to", self.ch_n.get())
    if self.dataset_n.get():
        self.plotting.configure(state='normal')
        self.clearplotting.configure(state='normal')
        self.simulateplotting.configure(state='normal')

    
def Subj_changed(self):
    print("Subject updated to", self.dataset_n.get())
    if self.dataset_n.get():
        self.ch_n.configure(state='normal')
    
##################################################################################################
class MyWindow():
    def __init__(self, 
                 win,
                 model1,
                 model2,
                 model3,
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
        self.paramMenu.add_command(label="Open to see RF model parameters", command=ModelParam1)
        self.paramMenu.add_command(label="Open to see SVC model parameters", command=ModelParam2)
        self.paramMenu.add_command(label="Open to see LSTM model parameters", command=ModelParam3)
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
        n_dataset = tk.StringVar()
        self.dataset_n = Combobox(self.frame1, textvariable=n_dataset)
        self.dataset_n['values'] = ['subject #' + str(x) for x in range(len(data))]
        self.dataset_n.grid(row=1,column=1,pady=5)
        self.dataset_n.bind("<<ComboboxSelected>>", lambda event: Subj_changed(self))
        
        self.ch_label = Label(self.frame1, text = '')
        self.ch_label.grid(row=2,column=0,pady=5)
        n_channel = tk.StringVar()
        self.ch_n = Combobox(self.frame1, textvariable=n_channel, state='disable')
        self.ch_n['values'] = ['channel ' + str(x) for x in chnames]
        self.ch_n.grid(row=2,column=1,pady=5)
        self.ch_n.bind("<<ComboboxSelected>>", lambda event: ch_changed(self))
        
        ####################################################################################
        
        self.fig = Figure(figsize=(14,6.5))
        
        self.ax = self.fig.subplot_mosaic([['TopLeft', 'Right'],['BottomLeft','Right']],
                                  gridspec_kw={'width_ratios':[10, 5]})
        self.ax2 = self.ax['TopLeft'].twinx()
        
        self.ax['TopLeft'].set_title('Raw Data')
        self.ax['TopLeft'].set_ylabel('Signal (Arb. Unit)')
        self.ax['TopLeft'].set_xlabel('Seconds')
        self.ax2.set_ylim(0,2)
        self.ax2.set_ylabel('events (1 or 2)')

        self.ax['BottomLeft'].set_title('Power Spectral Density')
        self.ax['BottomLeft'].set_ylabel('Signal (Arb. Unit)')
        self.ax['BottomLeft'].set_xlabel('Frequency (Hz)')
        
        self.ax['Right'].set_title('Spectral embedding with covariances')
        self.ax['Right'].set_ylabel('')
        self.ax['Right'].set_xlabel('')
        
        self.fig.tight_layout()
        
        self.output = FigureCanvasTkAgg(self.fig, master=self.frame2)
        self.output.draw()
        self.output.get_tk_widget().pack()
        
        ####################################################################################
        
        self.plotting = Button(self.frame1, text='Plot', command=lambda: Plot(self,
                                                                            data,
                                                                            ),
                               state='disabled')
        self.plotting.grid(row=2,column=2,pady=5)
        
        self.clearplotting = Button(self.frame1, text='Clear Plot', command=lambda: Clearplot(self,), state='disabled')
        self.clearplotting.grid(row=2,column=3,pady=5)
        
        self.simulateplotting = Button(self.frame1,text='Simulation',command=lambda: simulate(self,
                                                                                           win,
                                                                                           data,
                                                                                           model1,
                                                                                           model2,
                                                                                           model3
                                                                                           ),
                                       state='disabled')
        #self.insert_lig.place(x=10,y=180)
        self.simulateplotting.grid(row=2,column=4,pady=5)
        
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
                       start.model3,
                       start.data,
                       )
        window.title('RTI-simulator')
    
        window.eval('tk::PlaceWindow . center')   
        
        window.quit()
        
        window.mainloop()
    except:
        pass
    
    
    
    
    
    
    
    
    
