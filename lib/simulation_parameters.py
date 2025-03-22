# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import argparse

from .functions import *
from .learnable_parameters import *

#Computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######## Simulation Parameters selected by user ########
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gainphase", type=int, default=0,
                    help="Binary flag controlling whether we consider gain-phase impairments (0 for no impairments and 1 otherwise).")
parser.add_argument("-s", "--seed", type=int, default=10,
                    help="Integer controling the seed to use in the simulations.")
parser.add_argument("-l", "--loss", type=int, default=0,
                    help="Integer controling the loss function to use during training (0 for 'max' and 1 for 'reconstruction').")
args = parser.parse_args()

sim_seed = args.seed

#Fix seed for reproducibility
torch.manual_seed(sim_seed)
np.random.seed(sim_seed)
torch.cuda.manual_seed(sim_seed)
#Try to make results reproducible
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
#To use torch.use_deterministic_algorithms(True), we need the following sentences
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8" #This will increase memory, more info: https://docs.nvidia.com/cuda/cublas/#results-reproducibility

######## Simulation Parameters ########
save_path      = 'results/'                                                             #Directory to save results
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
gain_imp_flag  = args.gainphase                                                         #True to include gain-phase impairments, False otherwise
N              = 64                                                                     #Number of Rx antennas
S              = 256                                                                    #Number of subcarriers
delta_f        = torch.tensor(240e3, dtype=torch.float32, device=device)                #Spacing between subcarriers
fc             = torch.tensor(60e9, dtype=torch.float32, device=device)                 #Carrier frequency [Hz]
lambd          = 3e8 / fc                                                               #Wavelength [m]
boltzmann      = torch.tensor(1.38e-23, device=device)                                  #Boltzmann constant [J/K]
sys_temp       = torch.tensor(290, device=device)                                       #System temperature [K]
N0             = 10*torch.log10(boltzmann*sys_temp/1e-3)                                #Noise PSD [dBm/Hz]
noiseFigure    = torch.tensor(0, device=device)                                         #Noise Figure [dB]
noiseVariance  = 10**((N0+noiseFigure)/10)*1e-3*S*delta_f                               #Receiver noise variance [W]
SNR_sens_dB    = 15                                                                     #Sensing SNR [dB]
var_gain       = 10**(SNR_sens_dB/10)*noiseVariance/(N*S)                               #Variance of the complex channel gain
msg_card       = 4                                                                      #Comm. constellation size
target_pfa     = 1e-2                                                                   #Target false alarm prob. for ISAC
delta_pfa      = 1e-3                                                                   #Max deviation from target_pfa
range_min_glob = torch.tensor(10, dtype=torch.float32, device=device)                   #Minimum possible considered range of the targets
range_max_glob = torch.tensor(43.75, dtype=torch.float32, device=device)                #Maximum possible considered range of the targets
angle_min_glob = torch.tensor(-np.pi/2, dtype=torch.float32, device=device)             #Minimum possible considered angle of the targets
angle_max_glob = torch.tensor(np.pi/2, dtype=torch.float32, device=device)              #Maximum possible considered angle of the targets
Ngrid_angle    = 100                                                                    #Number of points in the oversampled grid of angles
Ngrid_range    = 100                                                                    #Number of points in the oversampled grid of ranges
refConst       = MPSK(msg_card, np.pi/4, device=device)                                 #Refence constellation (PSK)
angle_res      = 2/N                                                                    #Angle resolution (roughly) [rad]
range_res      = 3e8 / (2*S*delta_f)                                                    #Range resolution (roughly) [m]
pixels_angle   = int(angle_res / (np.pi/Ngrid_angle))                                   #Pixels corresponding to the angle resolution
pixels_range   = int(range_res / ((range_max_glob - range_min_glob)/Ngrid_range))       #Pixels corresponding to the range resolution
Tcp            = 0.07/delta_f                                                           #Cyclic prefix time [s]
Rcp            = Tcp*3e8                                                                #Equivalent range corresponding to Tcp [m]

# Generate a new phase-impairment realization so that we can average the performance for different realizations
gain = torch.rand(N,1, device=device, dtype=torch.cfloat)*0.1 + 0.95        #Gain in [0.95, 1.05]
phase = torch.rand(N,1, device=device, dtype=torch.cfloat)*np.pi - np.pi/2  #Phase in [-pi/2, pi/2]
gain_phase = gain*torch.exp(1j*phase)
gain_phase *= np.sqrt(N)/torch.norm(gain_phase)     #Normalize impairments such that the squared norm is equal to N
print(f'The considered impairments are \n {gain_phase}', flush=True)
#We allow for training of random [theta_min, theta_max] and [range_min, range_max]
theta_mean_min = torch.tensor(-60*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max = torch.tensor(60*np.pi/180, dtype=torch.float32, device=device)
theta_span_min = torch.tensor(10*np.pi/180, dtype=torch.float32, device=device)
theta_span_max = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
range_mean_min = (range_min_glob + range_max_glob)/2.0           #This fixes the target range sector to [range_min_glob, range_max_glob]
range_mean_max = (range_min_glob + range_max_glob)/2.0
range_span_min = range_max_glob - range_min_glob
range_span_max = range_max_glob - range_min_glob
batch_size     = 1024
#Create an oversampled dictionary of possible target ranges and angles
angle_grid = torch.linspace(angle_min_glob, angle_max_glob, Ngrid_angle, device=device)
range_grid = torch.linspace(range_min_glob, range_max_glob, Ngrid_range, device=device)
#Testing values after training
nTestSamples         = int(1e6)                      #This value will be slightly changed to be a multiple of batch_size
numTestIt            = nTestSamples // batch_size     
nTestSamples         = numTestIt*batch_size       
theta_mean_min_sens_test = torch.tensor(-60*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max_sens_test = torch.tensor(60*np.pi/180, dtype=torch.float32, device=device)
theta_span_min_sens_test = torch.tensor(10*np.pi/180, dtype=torch.float32, device=device)
theta_span_max_sens_test = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
range_mean_min_sens_test = (range_min_glob + range_max_glob)/2.0
range_mean_max_sens_test = (range_min_glob + range_max_glob)/2.0
range_span_min_sens_test = range_max_glob - range_min_glob
range_span_max_sens_test = range_max_glob - range_min_glob

######## Other Parameters computed from simulation parameters ########
#Assumed gain-phase vector
if gain_imp_flag:
    assumed_gain_phase = torch.ones((N,1), dtype=torch.cfloat, device=device)
else: #Known impairments
    assumed_gain_phase = torch.clone(gain_phase)        
    
######## NN-related parameters ########
initial_phase_gain = torch.clone(assumed_gain_phase)
network_impairment = ImpairmentNet(initial_phase_gain.cpu()).to(device)
#Define different learning rates and optimizers for the case of sequential training
learning_rate           = 1e-2
optimizer = torch.optim.Adam(list(network_impairment.parameters()), lr = learning_rate)
scheduler_flag          = False                                                           #Flag to use scheduler if True or skip it if False
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.1)
loss_list               = ['max', 'reconstruction']              #List of loss functions to use during training
loss_training           = loss_list[args.loss]
train_it                = int(1e4)                               #Number of training iterations
test_iterations_list    = np.linspace(1,train_it,10).astype(int) #Iterations where to test the network performance during training   
thresholds_roc          = torch.logspace(-2.88,-2.15,10)         #Thresholds to compute the ROC curve
thresholds_pfa          = torch.logspace(-2.39, -2.35, 3)        #Thresholds to test the network for a Pfa of 10^(-2)