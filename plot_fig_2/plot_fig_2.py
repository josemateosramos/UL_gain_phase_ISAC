# -*- coding: utf-8 -*-
"""
# Functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def convertAngleRangeToPos(angle, range):
    '''
    Function that computes position based on the angle and range.
    Inputs can have arbitrary shapes as long as they have the same shape,
    but the output will have shape (-1,2)
    '''
    x_coord = range*torch.cos(angle)
    y_coord = range*torch.sin(angle)

    pos = torch.cat((x_coord.reshape(-1,1), y_coord.reshape(-1,1)), dim=1)

    return pos

def generateUniform(min_val, max_val, out_shape, device='cuda'):
    '''Function that generates a tensor of shape out_shape with uniformly
    distributed values between min_val and max_val.
    Inputs:
        - min_val: minimum value of the uniform distribution.
        - max_val: maximum value of the uniform distribution. It should have the
        same shape as min_val.
        - out_shape: output shape. List of integers.
        - device: device on which to perform operations ('cpu' or 'cuda').
        Defult 'cpu'.
    Output
        - output: tensor whose entries are uniformly distributed between min_val
        and max_val.
    '''
    return torch.rand(out_shape, device=device) * (max_val - min_val) + min_val

def generateInterval(mean_min, mean_max, span_min, span_max, batch_size=1, device='cuda'):
    '''
    Function that creates an interval [minimum, maximum] as
    [minimum, maximum] = mean + [span/2, -span/2], where
    mean~U(mean_min, mean_max), and span~U(span_min, span_max).
    Inputs:
        - mean_min: minimum value of mean. Real number.
        - mean_max: maximum value of mean. Real number.
        - span min: minimum value of span. Real number.
        - span_max: maximum value of span. Real number.
        - batch_size: number of intervals we want to randomly draw. Integer.
        - device: 'cpu' or 'cuda'.
    Outputs:
        - minimum: minimum value of the intervals. Shape: (batch_size,1).
        - maximum: maximum value of the intervals. Shape: (batch_size,1).
    '''
    mean = generateUniform(mean_min, mean_max, (batch_size, 1), device=device)
    span = generateUniform(span_min, span_max, (batch_size, 1), device=device)
    minimum = mean - span / 2.0
    maximum = mean + span / 2.0

    return minimum, maximum

def steeringMatrix(theta, N, spacing, lambd, gp_error=None):
    '''Function that returns a matrix whose columns are steering vectors of the
    form e^(-j*2*pi/lamb*pos*sin(theta)), where theta~U(theta_min, theta_max).
    The uniform distribution is applied element-wise.
    Inputs:
        - theta:
        - N: number of antenna elements. Real number.
        - gp_error.
    Outputs:
        - matrix: matrix whose columns are steering vectors.
        - theta: realization of the random variable of the angle.
    '''
    device = theta.device
    if gp_error == None:
        gp_error = torch.ones((N,1), dtype=torch.cfloat, device=device)

    matrix = gp_error * torch.exp(-1j * 2 * np.pi * spacing / lambd * torch.arange(N, device=device).view(-1,1).to(torch.cfloat) @ torch.sin(theta.type(torch.cfloat)))
    return matrix


def delayMatrix(r_tgt, delta_f, S):
    '''Function that returns a matrix whose columns are complex vectors
    of the form e^(-j*2*pi*s*delta_f*2*R/c), where s=0,...,S-1 and
    theta~U(theta_min, theta_max). The uniform distribution is applied
    element-wise
    Inputs:
        - r_tgt: range of the target
        - delta_f: spacing between different subcarriers
        - S: number of subcarriers.
        - numColumns: number of columns of the output matrix.
    Outputs:
        - matrix: matrix whose columns are complex vectors.
        - range_tgt: realization of the random variable of the target range.
    '''
    device = r_tgt.device

    matrix = torch.exp(-1j*2*np.pi*delta_f * 2/3e8 * torch.arange(0,S, device=device).view(-1,1).type(torch.cfloat) @ r_tgt.type(torch.cfloat))
    return matrix

def MPSK(M, rotation=0, device='cuda'):
    '''Function that returns an array with all the symbols of a M-PSK
    constellation.
    Inputs:
        - M: size of the constellation
        - rotation: angle of rotation of the constellation [rad]'''
    return torch.exp(1j * (2*np.pi/M * torch.arange(0,M, device=device) + rotation))


def noise(var, dims, device='cuda'):
    '''Function that returns a complex vector that follows a complex Gaussian of
    zero mean and covariance matrix var*eye(dims[1]).
    Inputs:
        - var - variance of each component of the multivariate Gaussian
        - dims - dimensions of the output noise
            - dims[0] - #samples        - dims[1] - #dimensions
    Output:
        - noise_realization: realization of the complex Gaussian random variable
    '''

    return torch.sqrt(var/2.0) * (torch.randn(dims, device=device) + 1j * torch.randn(dims, device=device))

def createBSMatrix(theta, N, S, delta_f, spacing, device='cuda'):
    '''
    Function that creates the beam-squinting matrix C
    '''
    s_arange = torch.arange(S, dtype=torch.float32, device=device).view(1,-1)
    tau_arange = torch.arange(N, dtype=torch.float32, device=device).view(-1,1) * spacing * torch.sin(theta)/3e8
    return torch.exp(-1j*2*np.pi*delta_f*(tau_arange @ s_arange))

def radarCH(target, var_gamma, N, theta, range_tgt, spacing, lambd, delta_f, comm_sym, var_noise, gp_error, bs_imp_flag, device='cuda'):
    '''
    Function that computes the observations from the radar channel
    target: binary (batch_size, 1, 1)
    var_gamma: real
    N: integer
    theta_min, range_tgt: (batch_size,1,1)
    spacing: real
    lambd: real
    comm_sym: (batch_size, S, 1)
    var_nosie: real
    gp_error: gain-phase error, (N,1)
    '''
    #Extract data from inputs
    batch_size, S, _ = comm_sym.shape

    #Create random variables
    gamma = noise(var_gamma, (batch_size,1,1), device=device)
    rec_noise = noise(var_noise, (batch_size,N,S), device=device)

    #Create steering vectors
    arx = steeringMatrix(theta, N, spacing, lambd, gp_error)     #Shape: (batch_size,N,1)
    b_tau = delayMatrix(range_tgt, delta_f, S)               #Shape: (batch_size,S,1)

    #Create BS matrices
    if bs_imp_flag:
        CR = createBSMatrix(theta, N, S, delta_f, spacing, device=device)      #Shape: (batch_size,N,S)
    else:
        CR = torch.ones((batch_size, N, S), dtype=torch.cfloat, device=device)

    #Compute noise-free observation received observations
    Y_noisefree = target * gamma * (arx @ (b_tau * comm_sym).transpose(-1,-2)) * CR

    return Y_noisefree + rec_noise

"""# Simulation paramters"""

batch_size     = 100
gain_imp_flag  = 0      ###### This is where the gain-phase errors can be changed and the angle-delay map

"""## Other simulations parameters"""
#Fix seed for reproducibility
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)
#Computation device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######## Simulation Parameters ########
save_path      = 'results/'                                                             #Directory to save results
device         = 'cuda' if torch.cuda.is_available() else 'cpu'
bs_imp_flag    = 0                                                     #True to include beam squinting impairments, False otherwise
N              = 64                                                                     #Number of Rx antennas
S              = 256                                                                   #Number of subcarriers
delta_f        = torch.tensor(240e3, dtype=torch.float32, device=device)                #Spacing between subcarriers
fc             = torch.tensor(60e9, dtype=torch.float32, device=device)                 #Carrier frequency [Hz]
lambd          = 3e8 / fc                                                               #Wavelength [m]
boltzmann      = torch.tensor(1.38e-23, device=device)                                  #Boltzmann constant [J/K]
sys_temp       = torch.tensor(290, device=device)                                       #System temperature [K]
N0             = 10*torch.log10(boltzmann*sys_temp/1e-3)                                #Noise PSD [dBm/Hz]
noiseFigure    = torch.tensor(0, device=device)                                         #Noise Figure [dB]
noiseVariance  = 10**((N0+noiseFigure)/10)*1e-3*S*delta_f                               #Receiver noise variance [W]
SNR_sens_dB    = 15                                                                     #Sensing SNR [dB]
var_gain       = 10**(SNR_sens_dB/10)*noiseVariance/(N*S)                         #Variance of the complex channel gain
msg_card       = 4                                                                      #Comm. constellation size
target_pfa     = 1e-2                                                                   #Target false alarm prob. for ISAC
delta_pfa      = 1e-3                                                                   #Max deviation from target_pfa
range_min_glob = torch.tensor(10, dtype=torch.float32, device=device)                   #Minimum possible considered range of the targets
range_max_glob = torch.tensor(43.75, dtype=torch.float32, device=device)                                           #Maximum possible considered range of the targets
angle_min_glob = torch.tensor(-np.pi/2, dtype=torch.float32, device=device)             #Minimum possible considered angle of the targets
angle_max_glob = torch.tensor(np.pi/2, dtype=torch.float32, device=device)              #Maximum possible considered angle of the targets
Ngrid_angle    = 1000                                                                     #Number of points in the oversampled grid of angles
Ngrid_range    = 1000                                                                     #Number of points in the oversampled grid of ranges
refConst       = MPSK(msg_card, np.pi/4, device=device)                                 #Refence constellation (PSK)
angle_res      = 2/N                                                                    #Angle resolution (roughly) [rad]
range_res      = 3e8 / (2*S*delta_f)                                                    #Range resolution (roughly) [m]
pixels_angle   = int(angle_res / (np.pi/Ngrid_angle))                                   #Pixels corresponding to the angle resolution
pixels_range   = int(range_res / ((range_max_glob - range_min_glob)/Ngrid_range))       #Pixels corresponding to the range resolution
Tcp            = 0.07/delta_f                                                           #Cyclic prefix time [s]
Rcp            = Tcp*3e8                                                                #Equivalent range corresponding to Tcp [m]

# Phase and gain errors
#Here is the exact realization used in this script
gain_phase = torch.tensor([0.9319+0.3492j, 0.1193+0.9905j, 0.7468+0.6355j, 0.7582-0.6685j,
         0.4612-0.8542j, 0.3416+0.9297j, 0.0665+1.0165j, 0.7688-0.7096j,
         0.5418-0.8537j, 0.8376+0.5133j, 0.4344-0.9053j, 0.9228-0.3687j,
         0.0175-1.0196j, 0.9137+0.4187j, 0.8762-0.5200j, 0.6685-0.8068j,
         0.5297-0.8228j, 0.8490-0.5237j, 0.7423+0.6861j, 0.9950+0.0887j,
         0.2152+0.9710j, 0.8085-0.5085j, 0.8570-0.4242j, 0.0318+0.9538j,
         0.0688-0.9479j, 0.7764-0.5523j, 0.5400-0.7955j, 0.0567+0.9526j,
         0.6384+0.7069j, 0.8830-0.5130j, 0.8093-0.6370j, 0.9155+0.4786j,
         0.8849-0.5197j, 0.8978+0.4762j, 0.7086-0.7355j, 0.5477-0.8027j,
         1.0435-0.0410j, 0.6015+0.7937j, 0.2406+1.0143j, 0.9541+0.3452j,
         0.6184-0.8342j, 0.7080+0.7606j, 0.8092+0.5684j, 0.7719+0.6116j,
         0.8719-0.5035j, 0.9749-0.2098j, 0.7539+0.6678j, 0.8164+0.6192j,
         0.4833+0.8298j, 0.5863-0.8065j, 0.5297+0.8007j, 0.8581-0.6010j,
         0.9634-0.0380j, 0.7455-0.6798j, 0.8807-0.4426j, 0.5033+0.8960j,
         0.2268+0.9734j, 0.8492+0.5172j, 0.3962-0.9571j, 0.8745-0.5721j,
         0.4457+0.8537j, 0.8670-0.3911j, 0.2847-0.9623j, 0.9483+0.3622j], dtype=torch.cfloat, device=device).reshape(-1,1)
#We allow for training of random [theta_min, theta_max] and [range_min, range_max]
theta_mean_min = torch.tensor(-60*np.pi/180, dtype=torch.float32, device=device)
theta_mean_max = torch.tensor(60*np.pi/180, dtype=torch.float32, device=device)
theta_span_min = torch.tensor(10*np.pi/180, dtype=torch.float32, device=device)
theta_span_max = torch.tensor(20*np.pi/180, dtype=torch.float32, device=device)
range_mean_min = (range_min_glob + range_max_glob)/2.0           #This fixes the target range sector to [range_min_glob, range_max_glob]
range_mean_max = (range_min_glob + range_max_glob)/2.0
range_span_min = range_max_glob - range_min_glob
range_span_max = range_max_glob - range_min_glob
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

#Assumed gain-phase vector
if gain_imp_flag:
    assumed_gain_phase = torch.ones((N,1), dtype=torch.cfloat, device=device)
else: #Known impairments
    assumed_gain_phase = torch.clone(gain_phase)

"""# Compute received observation"""

#Generate random number of targets in the scene
target = torch.ones((batch_size, 1, 1), device=device)

#Generate random angles and ranges for each batch sample.
theta_min, theta_max = generateInterval(theta_mean_min_sens_test, theta_mean_max_sens_test, theta_span_min_sens_test, theta_span_max_sens_test, batch_size, device=device)
theta_sample = generateUniform(theta_min, theta_max, theta_min.shape, device=device)
theta_sample = theta_sample.reshape(batch_size,1,1)
range_min, range_max = generateInterval(range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, batch_size, device=device)
range_sample = generateUniform(range_min, range_max, range_min.shape, device=device)
range_sample = range_sample.reshape(batch_size,1,1)
#True positions from true angle and range
pos_sample = convertAngleRangeToPos(theta_sample, range_sample)

#Generate complex symbols
msg = torch.randint(0,msg_card,size=(batch_size*S,), dtype=torch.int64, device=device)
comm_sym = refConst[msg].reshape(batch_size, S, 1)

#Compute observations
Y_obs = radarCH(target, var_gain, N, theta_sample, range_sample, lambd/2.0, lambd, delta_f, comm_sym, noiseVariance, gain_phase, bs_imp_flag, device=device)

"""## Compute angle-delay map"""

device = 'cpu'
spacing = lambd/2.0
gp_vector = assumed_gain_phase

batch_size, N, S = Y_obs.shape

#Retrieve number of angular and range sectors that are being considered (either 1 or batch_size)
num_angle_sectors = 1 if ((not torch.is_tensor(theta_min)) or (theta_min.dim()==0)) else len(theta_min)
num_angle_sectors_arange = torch.arange(num_angle_sectors, device=device)
num_range_sectors = 1 if ((not torch.is_tensor(range_min)) or (range_min.dim()==0)) else len(range_min)
num_range_sectors_arange = torch.arange(num_range_sectors, device=device)

#Compute matrices of steering vectors and delays so that only the uncertainty region is taken into account
delta_angle = (theta_max-theta_min)/(Ngrid_angle-1)
theta_grid_matrix = theta_min + delta_angle*torch.arange(Ngrid_angle, device=device).reshape(1,-1)  #Shape: (batch_size, Ngrid_angle)
theta_grid_vector = theta_grid_matrix.reshape(1,-1)                                                 #Shape: (1, batch_size*Ngrid_angle)
A_rx = steeringMatrix(theta_grid_vector, N, spacing, lambd, gp_vector)                   #Shape: (N, batch_size*Ngrid_angle)
A_rx = A_rx.transpose(-1,-2).reshape(batch_size, Ngrid_angle, N)                     #(already transposed for AD map)

delta_range = (range_max-range_min)/(Ngrid_range-1)
range_grid_matrix = range_min + delta_range*torch.arange(Ngrid_range, device=device).reshape(1,-1)  #Shape: (batch_size, Ngrid_range)
range_grid_vector = range_grid_matrix.reshape(1,-1)                                                 #Shape: (1, batch_size*Ngrid_range)
delay_matrix = delayMatrix(range_grid_vector, delta_f, S)                     #Shape: (S, batch_size*Ngrid_range)
delay_matrix = delay_matrix.transpose(-1,-2).reshape(batch_size, Ngrid_range, S).permute(0,2,1)     #Shape: (batch_size, S, Ngrid_range)
delay_matrix *= comm_sym                                                                            #Consider the effect of communication symbols

#Compute the angle-delay map
ad_map = torch.abs(A_rx.conj() @ Y_obs @ delay_matrix.conj())**2        #(batch_size, Ngrid_angle, Ngrid_range)

"""## Represent AD map"""

index = 26
image = ad_map[index, :, :]

print('The max of the ad_map is: ', torch.max(image))

if not gain_imp_flag:
    image /= torch.max(image)
else:
    image /= 9.5793e-08         #This is the max value under no impairments

image_dB = 10*np.log10(image.numpy())

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Enable LaTeX fonts
rc('font',**{'family':'serif'})

# Get a list of available colormaps
colormaps = [m for m in cm.datad]

plt.figure(figsize=(8,8))
im = plt.imshow(image_dB, aspect=2, cmap=colormaps[52], interpolation=None, extent=[10, 43.75, theta_max[index].item()*180/np.pi, theta_min[index].item()*180/np.pi])
plt.scatter(range_sample[index], theta_sample[index]*180/np.pi, color=(180/255, 73/255, 1), marker='s', s=60, label='True target position')
cbar = plt.colorbar(im, shrink=0.6)
im.set_clim(vmin=-15, vmax=0)
cbar.set_label(r'[dB]', fontsize=16)
cbar.ax.tick_params(labelsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel(r'Range [m]', fontsize=16)
plt.ylabel(r'Angle [deg]', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),  fancybox=True, fontsize=16)
#### Save image
if not gain_imp_flag:
    plt.savefig('angle_range_map_ideal.png', format='png', bbox_inches="tight", dpi=500)
else:
    plt.savefig('angle_range_map_impairments.png', format='png', bbox_inches="tight", dpi=500)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Enable LaTeX fonts
rc('font',**{'family':'serif'})

# Get a list of available colormaps
colormaps = [m for m in cm.datad]

plt.figure(figsize=(12,10))
im = plt.imshow(image_dB, aspect=2, cmap=colormaps[52], interpolation=None, extent=[10, 43.75, theta_max[index].item()*180/np.pi, theta_min[index].item()*180/np.pi])
plt.scatter(range_sample[index], theta_sample[index]*180/np.pi, color=(180/255, 73/255, 1), marker='s', s=60, label='True target position')
cbar = plt.colorbar(im, shrink=0.6,location='left', pad=0.12)
im.set_clim(vmin=-15, vmax=0)
cbar.set_label(r'[dB]', fontsize=16, rotation=0, labelpad=-50,y=-.08)
cbar.ax.tick_params(labelsize=16)
plt.tick_params(axis='both', which='major', labelsize=16)
plt.xlabel(r'Range [m]', fontsize=16)
plt.ylabel(r'Angle [deg]', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),  fancybox=True, fontsize=16)
#### Save image
if not gain_imp_flag:
    plt.savefig('angle_range_map_ideal.png', format='png', bbox_inches="tight", dpi=500)
else:
    plt.savefig('angle_range_map_impairments.png', format='png', bbox_inches="tight", dpi=500)


print(theta_sample[index]*180/np.pi)
print(range_sample[index])
