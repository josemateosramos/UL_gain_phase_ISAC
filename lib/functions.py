# -*- coding: utf-8 -*-
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
    '''Function that returns a steering vector of the form e^(-j*2*pi/lamb*pos*sin(theta)).
    Inputs:
        - theta: angle to evaluate. Tensor of shape (batch_size,1,1) or (batch_size,1,N).
        In the latter case the output will be a matrix whose columns are steering vectors.
        - N: number of antenna elements. Integer
        - spacing: spacing between antenna elements. Float.
        - lambd: wavelength. Float.
        - gp_error: gain-phase errors. Complex tensor of shape (N,1).
    Outputs:
        - matrix: matrix whose columns are steering vectors.
    '''
    device = theta.device
    if gp_error == None:
        gp_error = torch.ones((N,1), dtype=torch.cfloat, device=device)

    matrix = gp_error * torch.exp(-1j * 2 * np.pi * spacing / lambd * torch.arange(N, device=device).view(-1,1).to(torch.cfloat) @ torch.sin(theta.type(torch.cfloat)))
    return matrix


def delayMatrix(r_tgt, delta_f, S):
    '''Function that returns a complex delay vector of the form e^(-j*2*pi*s*delta_f*2*R/c), 
    where s=0,...,S-1.
    Inputs:
        - r_tgt: range of the target. Tensor of shape (batch_size,1,1) or (batch_size,1,N)
        In the latter case the output will be a matrix whose columns are delay vectors.
        - delta_f: spacing between different subcarriers. Float
        - S: number of subcarriers. Integer.
    Outputs:
        - matrix: matrix whose columns are complex delay vectors.
    '''
    device = r_tgt.device

    matrix = torch.exp(-1j*2*np.pi*delta_f * 2/3e8 * torch.arange(0,S, device=device).view(-1,1).type(torch.cfloat) @ r_tgt.type(torch.cfloat))
    return matrix

def MPSK(M, rotation=0, device='cuda'):
    '''Function that returns an array with all the symbols of a M-PSK
    constellation.
    Inputs:
        - M: size of the constellation. Integer.
        - rotation: angle of rotation of the constellation [rad]. Float
    Output:
        - Complex tensor of shape (M,) containing the M complex symbols of 
        the MPSK constellation  
    '''
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

def radarCH(target, var_gamma, N, theta, range_tgt, spacing, lambd, delta_f, comm_sym, var_noise, gp_error, device='cuda'):
    '''
    Function that computes the signal received at the sensing receiver
    Inputs:
        - target: binary tensor representing whether there is a target in the environment or not.
        Shape: (batch_size, 1, 1).
        - var_gamma: variance of the complex channel gain. Float.
        - N: number of antenna elements in the sensing receiver. Integer.
        - theta: angle of the target. Float tensor of shape (batch_size,1,1)
        - range_tgt: range of the target. Float tensor of shape (batch_size,1,1)
        - spacing: spacing between different antenna elements in the 
        array of the sensing receiver. Float
        - lambd: wavelength. Float.
        - comm_sym: communication symbols. Complex tensor of shape (batch_size, S, 1).
        - var_noise: variance of the noise at the sensing receiver. Float.
        - gp_error: vector representing the gain-phase errors. Complex tensor 
        of shape (N,1).
    Outputs:
        - Complex tensor of shape (batch_size, N, S) representing the signal received
        at the sensing receiver.
    '''
    #Extract data from inputs
    batch_size, S, _ = comm_sym.shape

    #Create random variables
    gamma = noise(var_gamma, (batch_size,1,1), device=device)
    rec_noise = noise(var_noise, (batch_size,N,S), device=device)

    #Create steering vectors
    arx = steeringMatrix(theta, N, spacing, lambd, gp_error)     #Shape: (batch_size,N,1)
    b_tau = delayMatrix(range_tgt, delta_f, S)                   #Shape: (batch_size,S,1)

    #Compute noise-free observation received observations
    Y_noisefree = target * gamma * (arx @ (b_tau * comm_sym).transpose(-1,-2)) 

    return Y_noisefree + rec_noise

######################
# Training functions #
######################

def reconstructY(est_theta, est_range, Y_obs, comm_symb, noiseVariance, var_gain, N, spacing, lambd, gp_errors, delta_f):
    '''
    Function that estimates the complex channel gain and reconstructs the received signal from the angle and range estimates.
    This function is useful when choosing the reconstruction loss during training.
    Input:
        - est_theta: estimated angle. Real tensor of shape (batch_size, )
        - est_range: estimated range. Real tensor of shape (batch_size, )
        - Y_obs: signal received at the sensing receiver. Complex tensor of shape (batch_size, N, S).
        - comm_symb: transmitted communication symbols. Complex tensor of shape (batch_size,)
        - noiseVariance: variance of the noise at the sensing receiver. Float.
        - var_gain: variance of the complex channel gain. Float.
        - N: number of antenna elements at the sensing receiver. Integer.
        - spacing: spacing between antenna elements in the sensing receiver. Float.
        - lambd: wavelength. Float.
        - gp_errors: vector representing the true gain-phase errors. Complex tensor of shape (N,1).
        - delta_f: subcarrier spacing. Float.
    Output: 
        - Reconstructed signal based on the estimated angle and range.
    '''
    #Infer data from inputs
    batch_size, N, S = Y_obs.shape

    #Construct steering vectors
    arx = steeringMatrix(est_theta, N, spacing, lambd, gp_errors)   #Shape: (batch_size, N, 1)
    brx = delayMatrix(est_range, delta_f, S)                        #Shape: (batch_size, S, 1)

    #Estimate gamma according to the MAPRT developed in the paper
    est_gamma = (arx.conj().permute(0,2,1) @ Y_obs @ (brx * comm_symb).conj()) / (N*torch.norm(comm_symb, dim=-2, keepdim=True)**2 + noiseVariance/var_gain)
    
    #Reconstruct received signal
    Y_reconst = est_gamma * (arx @ (brx * comm_symb).transpose(-2,-1))

    return Y_reconst

def trainNetworkRXUL(network, optimizer, var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                    theta_span_min_sens_test, theta_span_max_sens_test,
                    range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                    range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                    N, S, noiseVariance, delta_f, lambd, gain_phase, 
                    msg_card, refConst, threshold_list, batch_size, train_it, epoch_test_list, 
                    n_test_samples, target_pfa, delta_pfa, 
                    sch_flag=False, scheduler=None, loss='max', device='cuda'):
    '''
    Function to train the network that contains the gain-phase impairments.
    This function uses the maximum of the angle-delay map as loss function.
    Inputs: 
        - network: class containing the gain-phase impairments to optimize.
        - optimizer: class containing the optimizer to update the network parameters.
        - var_gain: variance of the complex channel gain. Float.
        - theta_mean_min_sens_test: minimum value of the mean of the angular sector for the target. Float.
        - theta_mean_max_sens_test: maximum value of the mean of the angular sector for the target. Float.
        - theta_span_min_sens_test: minimum value of the span of the angular sector for the target. Float.
        - theta_span_max_sens_test: maximum value of the span of the angular sector for the target. Float.
        - range_mean_min_sens_test: minimum value of the mean of the range sector for the target. Float.
        - range_mean_max_sens_test: maximum value of the mean of the range sector for the target. Float.
        - range_span_min_sens_test: minimum value of the span of the range sector for the target. Float.
        - range_span_max_sens_test: maximum value of the span of the range sector for the target. Float.
        - Ngrid_angle: number of points for the angle grid to compute the angle-delay map. Integer.
        - Ngrid_range: number of points for the range grid to compute the angle-delay map. Integer.
        - N: number of antenna elements in the sensing receiver. Integer.
        - S: number of subcarriers in the OFDM signal. Integer.
        - noiseVariance: variance of the noise at the sensing receiver.
        - delta_f: subcarrier spacing. Float.
        - lambd: wavelength. Float.
        - gain_phase: vector representing the true gain-phase errors. Complex tensor of shape (N,1).
        - msg_card: cardinality of the constellation to use in comms. Integer.
        - refConst: reference constellation (PSK). Complex tensor of shape (msg_card,)
        - threshold_list: list of thresholds to use to achieve the desired false alarm probability when
        testing the network.
        - batch_size: number of samples to use in a batch. Integer. 
        - train_it: number of training iterations. Integer.
        - epoch_test_list: list containing at which iterations the network should be tested. List of 
        integers.
        - n_test_samples: number of samples to use during testing. Integers.
        - target_pfa: target false alarm probability. Float.
        - delta_pfa: maximum allowable error with respect to target_pfa. Float.
        - sch_flag: true/false flag indicating whether to use a scheduler during training. True or False.
        - loss: loss function to use. String that can be either: 
            + 'max': use the negative of the maximum of the angle-delay map.
            + 'reconstruction': use the Frobenius norm between the received signal and the reconstructed one.
        - device: string indicating on which device operations should be performed. 'cpu' or 'cuda'.
    Outputs: 
        - loss_np: numpy array containing the values of the loss function for each iteration.
        Numpy array of length train_it. 
        - num_iterations: list containing the iterations where the network was tested. List of integers.
        - pd_test: detection probability for the iterations where the network was tested. List of floats.
        - pfa_test: false alarm probability for the iterations where the network was tested. 
        List of floats.
        - mse_angle_test: MSE of the target angle for the iterations where the network was tested. 
        List of floats. 
        - mse_range_test: MSE of the target range for the iterations where the network was tested. 
        List of floats.  
        - mse_pos_test: MSE of the target position for the iterations where the network was tested. 
        List of floats.  
    '''
    print('*** Training started ***', flush=True)
    network.train()

    #Create list to save loss function
    loss_np = []
    #Create lists to save testing results during training
    num_iterations, pd_test, pfa_test, mse_angle_test, mse_range_test, mse_pos_test = [], [], [], [], [], []

    for iteration in range(train_it):
        #Generate random number of targets in the scene
        target = torch.randint(0,2,(batch_size, 1, 1), dtype=torch.float32, device=device)

        #Generate random angles and ranges for each batch sample.
        theta_min, theta_max = generateInterval(theta_mean_min_sens_test, theta_mean_max_sens_test, theta_span_min_sens_test, theta_span_max_sens_test, batch_size, device=device)
        theta_sample = generateUniform(theta_min, theta_max, theta_min.shape, device=device)
        theta_sample = theta_sample.reshape(batch_size,1,1)
        range_min, range_max = generateInterval(range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test, range_span_max_sens_test, batch_size, device=device)
        range_sample = generateUniform(range_min, range_max, range_min.shape, device=device)
        range_sample = range_sample.reshape(batch_size,1,1)

        #Generate complex symbols
        msg = torch.randint(0,msg_card,size=(batch_size*S,), dtype=torch.int64, device=device)
        comm_sym = refConst[msg].reshape(batch_size, S, 1)

        #Compute observations (no channel backpropagation)
        with torch.no_grad():
            Y_obs = radarCH(target, var_gain, N, theta_sample, range_sample, lambd/2.0, lambd, delta_f, comm_sym, noiseVariance, gain_phase, device=device)

        #Estimate angle and range
        if loss == 'max':
            max_ad_map, est_angle, est_range = diffEstimatePos(Y_obs, comm_sym, theta_min, theta_max, range_min, range_max, Ngrid_angle, Ngrid_range, 
                                                        lambd/2.0, lambd, delta_f, network.impairments)
            loss = -torch.mean(max_ad_map)
        if loss == 'reconstruction':
            #Estimate angle and range using nondifferentiable approach (the gradient can still be computed from the recons. signal)
            _, est_angle, est_range = estimatePos(Y_obs, comm_sym, theta_min, theta_max, range_min, range_max, Ngrid_angle, Ngrid_range, 
                                                        lambd/2.0, lambd, delta_f, network.impairments)
            
            Y_reconst = reconstructY(est_angle.reshape(batch_size, 1, 1), est_range.reshape(batch_size, 1, 1), 
                                    Y_obs, comm_sym, noiseVariance, var_gain, N, lambd/2.0, lambd, network.impairments, delta_f)
            loss = torch.mean(torch.norm((Y_obs-Y_reconst).reshape(batch_size, N*S), dim=-1))
        
        #Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #Update numpy loss vectors
        loss_np.append(loss.item())
        if sch_flag:
            #Scheduler step
            scheduler.step()

        #Normalize impairments such that their squared norm is equal to N
        network.impairments.data = network.impairments.data * np.sqrt(N)/torch.norm(network.impairments.data)

        if (iteration+1) in epoch_test_list:
            network.eval()
            print(f'Testing training iteration {iteration+1}', flush=True)
            ### Test the network with the current number of iterations ###
            num_iterations.append(iteration+1)
            pd_temp, pfa_temp, mse_angle_temp, mse_range_temp, mse_pos_temp = \
                      testSensingFixedPfa(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                                        theta_span_min_sens_test, theta_span_max_sens_test,
                                        range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                                        range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                                        N, S, noiseVariance, delta_f, lambd, network.impairments, gain_phase, 
                                        msg_card, refConst, threshold_list, batch_size, n_test_samples, 
                                        target_pfa, delta_pfa,
                                        device=device)

            pd_test.append(pd_temp)
            pfa_test.append(pfa_temp)
            mse_angle_test.append(mse_angle_temp)
            mse_range_test.append(mse_range_temp)
            mse_pos_test.append(mse_pos_temp)
            network.train()
    return loss_np, num_iterations, pd_test, pfa_test, mse_angle_test, mse_range_test, mse_pos_test

########################
# Estimation functions #
########################
def estimatePos(Y_obs, comm_sym, theta_min, theta_max, range_min, range_max, Ngrid_angle, Ngrid_range, 
                    spacing, lambd, delta_f, gp_vector):
    '''
    Function to estimate the position of a target.
    Inputs:
        - Y_obs: Received signal at the sensing receiver. Complex tensor of size (batch_size, N, S)
        - comm_sym: sent communication symbols. Complex tensor of shape (batch_size, S, 1).
        - theta_min, theta_max: min and max values of the angular sector where the target can be.
        Real tensor of shape (batch_size, 1).
        - range_min, range_max: min and max values of the range sector where the target can be.
        Real tensor of shape (batch_size, 1).
        - Ngrid_angle: number of points to consider for the grid of angles. Integer.
        - Ngrid_range: number of points to consider for the grid of ranges. Integer.
        - spacing: distance between antenna elements in the sensing receiver. Float.
        - lambd: wavelength. Float.
        - delta_f: frequency spacing between subcarriers. Float. 
        - gp_vector: vector representing the estimated gain-phase errors. Complex tensor of shape (N,1).
    Outputs:
        - ad_map_max: maximum value of the angle-delay map. Real tensor of shape (batch_size,1,1) 
        - est_theta: estimated angle of the target. Real tensor of shape (batch_size, 1,1) 
        - est_range: estimated range of the target. Real tensor of shape (batch_size, 1,1)
    '''
    #Extract information from inputs
    device = theta_min.device
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
    #Multiply AD map so that we don't run into numerical issues
    ad_map *= 1e5
    #Estimate of AoA and range
    maximum = ad_map.reshape(batch_size, -1).argmax(dim=-1)   
    max_theta = torch.div(maximum, Ngrid_range, rounding_mode='floor')  #This return torch.int64 while torch.floor returns torch.float32 (// is deprecated)
    max_range = maximum % Ngrid_range                                   #Shape: (batch_size)
    #We can directly use previous tensors since they are integers
    est_theta = theta_grid_matrix[num_angle_sectors_arange, max_theta].view(batch_size, 1)
    est_range = range_grid_matrix[num_range_sectors_arange, max_range].view(batch_size, 1)
    # Compute maximum value of metric to calculate later probability of detection
    ad_map_max, _ = torch.max(ad_map.view(batch_size, -1), dim=1)

    return ad_map_max, est_theta, est_range


def diffEstimatePos(Y_obs, comm_sym, theta_min, theta_max, range_min, range_max, Ngrid_angle, Ngrid_range, 
                    spacing, lambd, delta_f, gp_vector):
    '''
    Differentiable function to estimate whether there is a target and its position.
    This function is to be used during training
    Inputs and outputs are described in the 'estimatePos' function above.
    '''
    #Extract information from inputs
    device = theta_min.device
    batch_size, N, S = Y_obs.shape

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
    # Compute maximum value of metric to calculate later probability of detection
    ad_map_max, _ = torch.max(ad_map.view(batch_size, -1), dim=1)
    #Normalize angle-delay map to the maximum of each batch_sample (multiplied by 100 so that softmax is not too smooth)
    ad_map *= 100/ad_map_max.reshape((batch_size, 1, 1))
        
    #Take softmax of AD map
    ad_map_soft = F.softmax(ad_map.reshape(batch_size, Ngrid_angle*Ngrid_range), dim=-1).reshape(batch_size, Ngrid_angle, Ngrid_range)
    #Perform weighted average of angles and ranges
    ad_sum_angle = torch.sum(ad_map_soft, dim=-1)
    est_theta = torch.sum(ad_sum_angle*theta_grid_matrix, dim=-1)
    ad_sum_range = torch.sum(ad_map_soft, dim=-2)
    est_range = torch.sum(ad_sum_range*range_grid_matrix, dim=-1)

    return ad_map_max, est_theta, est_range

                
########################
# Testing functions #
########################
def computePositionsTesting(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                            theta_span_min_sens_test, theta_span_max_sens_test,
                            range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                            range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                            N, S, noiseVariance, delta_f, lambd, assumed_gain_phase, gain_phase, 
                            msg_card, refConst, batch_size, numTestSamples, 
                            device='cuda'):
    '''
    Function that computes the position of the estimated target and the metric to later threshold for Pd and Pfa
    Inputs: 
        - var_gain: variance of the complex channel gain. Float.
        - theta_mean_min_sens_test: minimum value of the mean of the angular sector for the target. Float.
        - theta_mean_max_sens_test: maximum value of the mean of the angular sector for the target. Float.
        - theta_span_min_sens_test: minimum value of the span of the angular sector for the target. Float.
        - theta_span_max_sens_test: maximum value of the span of the angular sector for the target. Float.
        - range_mean_min_sens_test: minimum value of the mean of the range sector for the target. Float.
        - range_mean_max_sens_test: maximum value of the mean of the range sector for the target. Float.
        - range_span_min_sens_test: minimum value of the span of the range sector for the target. Float.
        - range_span_max_sens_test: maximum value of the span of the range sector for the target. Float.
        - Ngrid_angle: number of points for the angle grid to compute the angle-delay map. Integer.
        - Ngrid_range: number of points for the range grid to compute the angle-delay map. Integer.
        - N: number of antenna elements in the sensing receiver. Integer.
        - S: number of subcarriers in the OFDM signal. Integer.
        - noiseVariance: variance of the noise at the sensing receiver.
        - delta_f: subcarrier spacing. Float.
        - lambd: wavelength. Float.
        - gain_phase: vector representing the true gain-phase errors. Complex tensor of shape (N,1).
        - assumed_gain_phase: vector representing the assumed gain-phase errors. Complex tensor of shape (N,1).
        - msg_card: cardinality of the constellation to use in comms. Integer.
        - refConst: reference constellation (PSK). Complex tensor of shape (msg_card,)
        - batch_size: number of samples to use in a batch. Integer. 
        - numTestSamples: number of samples to use during testing. Integers.
        - device: string indicating on which device operations should be performed. 'cpu' or 'cuda'.
    Outputs: 
        - metric_save: maximum of the angle-delay map. Real tensor of shape (numTestSamples,) 
        - est_angle_save: estimated target angle. Real tensor of shape (numTestSamples,) 
        - est_range_save: estimated target range. Real tensor of shape (numTestSamples,) 
        - est_pos_save: estimated target position. Real tensor of shape (numTestSamples, 2) 
        - true_target: Binary vector indicating whether there was truly a target in the 
        scenario. Shape (numTestSamples,) 
        - true_angle: true target angle. Real tensor of shape (numTestSamples,) 
        - true_range: true target range. Real tensor of shape (numTestSamples,) 
        - true_pos: true target position. Real tensor of shape (numTestSamples, 2)
    '''
    iterations = numTestSamples // batch_size
    #Create lists to save results later on
    metric_save = torch.empty(numTestSamples)
    est_angle_save, est_range_save = torch.empty(numTestSamples), torch.empty(numTestSamples)
    est_pos_save = torch.empty((numTestSamples, 2))
    true_target = torch.empty(numTestSamples)
    true_angle, true_range = torch.empty(numTestSamples), torch.empty(numTestSamples)
    true_pos = torch.empty(numTestSamples, 2)

    with torch.no_grad():
        for i in range(iterations):
            #Generate random number of targets in the scene
            target = torch.randint(2,(batch_size, 1, 1), device=device)

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
            Y_obs = radarCH(target, var_gain, N, theta_sample, range_sample, lambd/2.0, lambd, delta_f, comm_sym, noiseVariance, gain_phase, device=device)

            #Estimate angle and range assuming no impairments
            metric, est_angle, est_range = estimatePos(Y_obs, comm_sym, theta_min, theta_max, range_min, range_max, Ngrid_angle, Ngrid_range, 
                                                                lambd/2.0, lambd, delta_f, assumed_gain_phase)
            est_pos = convertAngleRangeToPos(est_angle, est_range)

            #Save results
            metric_save[i*batch_size:(i+1)*batch_size] = metric.flatten()
            est_angle_save[i*batch_size:(i+1)*batch_size] = est_angle.flatten()
            est_range_save[i*batch_size:(i+1)*batch_size] = est_range.flatten()
            est_pos_save[i*batch_size:(i+1)*batch_size,:] = est_pos
            true_target[i*batch_size:(i+1)*batch_size] = target.flatten()
            true_angle[i*batch_size:(i+1)*batch_size] = theta_sample.flatten()
            true_range[i*batch_size:(i+1)*batch_size] = range_sample.flatten()
            true_pos[i*batch_size:(i+1)*batch_size,:] = pos_sample

            if ((i+1)%50)==0:
                print(f'Iteration {i+1} out of {iterations} iterations', flush=True)
        
    return metric_save, est_angle_save, est_range_save, est_pos_save, true_target, true_angle, true_range, true_pos


def getSensingMetrics(est_metric, true_target, threshold, est_angle=None, true_angle=None, 
                      est_range=None, true_range=None, est_pos=None, true_pos=None):
    '''
    Function that computes different metrics to assess the performance of the methods.
    Inputs:
        - est_metric: maximum of the angle-delay. Real tensor of shape (nTestSamples,).
        - true_target: binary vector representing there was a target in the scene.
        Shape: (nTestSamples,)
        - threshold: threshold to detect whether there is a target in the scene. Float.
        (The following metric can be None in case one wants to compute just prob. 
        detection and false alarm)
        - est_angle: estimated target angle. Real tensor of shape (nTestSamples,)
        - true_angle: true target angle. Real tensor of shape (nTestSamples,)
        - est_range: estimated target range. Real tensor of shape (nTestSamples,)
        - true_range: true target range. Real tensor of shape (nTestSamples,)
        - est_pos: estimated target position. Real tensor of shape (nTestSamples,2)
        - true_pos: true target position. Real tensor of shape (nTestSamples,)
    Outputs:
        - pd: probability of detection computed from the nTestSamples samples. Float.
        - pfa: false alarm probability computed from the nTestSamples samples. Float.
        - mse_angle: MSE between the true and estimated angles. Float
        - mse_range: MSE between the true and estimated ranges. Float 
        - mse_pos: MSE between the true and estimated positions. Float 
    '''
    #Get information from input data
    nTestSamples = len(true_target)

    #Threshold metric to estimate the number of targets
    est_target = (est_metric >= threshold)*1.0
    #Probability of correct detection
    pd = ( torch.minimum(est_target, true_target).sum()*1.0/true_target.sum() ).item()
    #Probability of false alarm
    pfa =  ( (torch.maximum(est_target, true_target)-true_target).sum()*1.0 \
            /(nTestSamples - true_target.sum()) ).item()

    if (est_pos==None) or (true_pos==None):
        mse_angle = mse_range = mse_pos = None
    else:
        #Compute MSEs when est_target=1 & true_target=1
        true_target_bool = true_target.to(torch.bool)
        est_target_bool = est_target.to(torch.bool)
        mask = true_target_bool & est_target_bool
        mse_angle = torch.mean(torch.abs(est_angle[mask] - true_angle[mask])**2).item()
        mse_range = torch.mean(torch.abs(est_range[mask] - true_range[mask])**2).item()
        mse_pos = torch.mean(torch.norm(est_pos[mask] - true_pos[mask], dim=-1)**2).item()

    return pd, pfa, mse_angle, mse_range, mse_pos


def testSensingROC(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                    theta_span_min_sens_test, theta_span_max_sens_test,
                    range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                    range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                    N, S, noiseVariance, delta_f, lambd, assumed_gain_phase, gain_phase, 
                    msg_card, refConst, threshold_list, batch_size, numTestSamples, 
                    device='cuda'):
    '''
    Function that test the sensing performance of the system to compute the ROC curve.
    Inputs:
        - var_gain: variance of the complex channel gain. Float.
        - theta_mean_min_sens_test: minimum value of the mean of the angular sector for the target. Float.
        - theta_mean_max_sens_test: maximum value of the mean of the angular sector for the target. Float.
        - theta_span_min_sens_test: minimum value of the span of the angular sector for the target. Float.
        - theta_span_max_sens_test: maximum value of the span of the angular sector for the target. Float.
        - range_mean_min_sens_test: minimum value of the mean of the range sector for the target. Float.
        - range_mean_max_sens_test: maximum value of the mean of the range sector for the target. Float.
        - range_span_min_sens_test: minimum value of the span of the range sector for the target. Float.
        - range_span_max_sens_test: maximum value of the span of the range sector for the target. Float.
        - Ngrid_angle: number of points for the angle grid to compute the angle-delay map. Integer.
        - Ngrid_range: number of points for the range grid to compute the angle-delay map. Integer.
        - N: number of antenna elements in the sensing receiver. Integer.
        - S: number of subcarriers in the OFDM signal. Integer.
        - noiseVariance: variance of the noise at the sensing receiver.
        - delta_f: subcarrier spacing. Float.
        - lambd: wavelength. Float.
        - assumed_gain_phase: vector representing the asummed gain-phase errors. Complex tensor of shape (N,1).
        - gain_phase: vector representing the true gain-phase errors. Complex tensor of shape (N,1).
        - assumed_gain_phase: vector representing the assumed gain-phase errors. Complex tensor of shape (N,1).
        - msg_card: cardinality of the constellation to use in comms. Integer.
        - refConst: reference constellation (PSK). Complex tensor of shape (msg_card,)
        - threshold_list: list of thresholds to use to achieve the desired false alarm probability when
        testing the network.
        - batch_size: number of samples to use in a batch. Integer. 
        - numTestSamples: number of samples to use during testing. Integers.
        - device: string indicating on which device operations should be performed. 'cpu' or 'cuda'.
    Outputs:
        - pd_save: detection probability for different thresholds. List of floats of 
        length len(threshold_list) 
        - pfa_save: false alarm probability for different thresholds. List of floats of 
        length len(threshold_list) 
        - angle_mse_save: MSE of the target angle for different thresholds. List of floats of 
        length len(threshold_list)
        - range_mse_save: MSE of the target range for different thresholds. List of floats of 
        length len(threshold_list)
        - pos_mse_save: MSE of the target position for different thresholds. List of floats of 
        length len(threshold_list)
    '''
    est_metric, est_angle, est_range, est_pos, \
        true_target, true_angle, true_range, true_pos \
             = computePositionsTesting(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                            theta_span_min_sens_test, theta_span_max_sens_test,
                            range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                            range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                            N, S, noiseVariance, delta_f, lambd, assumed_gain_phase, gain_phase, 
                            msg_card, refConst, batch_size, numTestSamples, device)

    #Create lists to save results
    pd_save, pfa_save, angle_mse_save, range_mse_save, pos_mse_save = [], [], [], [], []
    for t in threshold_list:
        pd, pfa, mse_angle, mse_range, mse_pos = getSensingMetrics(est_metric, true_target, t, est_angle, true_angle, 
                      est_range, true_range, est_pos, true_pos)
        pd_save.append(pd)
        pfa_save.append(pfa)
        angle_mse_save.append(mse_angle)
        range_mse_save.append(mse_range)
        pos_mse_save.append(mse_pos)

    return pd_save, pfa_save, angle_mse_save, range_mse_save, pos_mse_save

def obtainThresholdsFixedPfa(est_metric, true_tgt, target_pfa, delta_pfa, init_thr):
    '''
    Function that empirically estimates the thresholds to yield a target false
    alarm probability with a maximum allowable error.
    Since obtaining exactly pfa = target_pfa is very difficult, we compute 3
    pfa's, so that for any of those probabilities,
    target_pfa - delta_pfa < pfa < target_pfa + delta_pfa.
    We then linearly interpolate the results.
    Inputs:
        - est_metric: metric to threshold to estimate the number of targets.
        Float list of length nTestSamples.
        - true_presence: true number of targets per batch sample. Binary list
        of length nTestSamples.
        - target_pfa: target false alarm proability. Float.
        - delta_pfa: maximum allowable error for the target_pfa. Float.
        - init_thr: initial thresholds to start the algorithm. Float numpy array
        with more than 1 element (usually 3).
    Outputs
        - init_thr: final thresholds that achieve target_pfa - delta_pfa < pfa,
        pfa < target_pfa + delta_pfa.
    '''
    print(f'Initital thresholds were: {init_thr}', flush=True)
    device = est_metric.device

    with torch.no_grad():
        #Reset Pfa
        pfa = np.zeros((1,len(init_thr)))   #Set initial Pfa to enter loop
        while ((np.max(pfa) > target_pfa + delta_pfa) or (np.min(pfa) < target_pfa - delta_pfa)):
            #Lists to save final results
            pd, pfa = [], []

            #Compute detection and false alarm probabilities
            for t in range(len(init_thr)):
                pd_temp, pfa_temp, _, _, _ = getSensingMetrics(est_metric, true_tgt, init_thr[t])
                pd.append(pd_temp)
                pfa.append(pfa_temp)

            #Check if target_pfa - delta_pfa < pfa < target_pfa + delta_pfa. We use the std to update the threshold vector
            if target_pfa < np.min(pfa):
                init_thr += torch.std(init_thr)
                # print(f'Target Pfa is lower than any Pfa, current pfa: {pfa}, current thresholds (*1e-3): {init_thr*1e3}', flush=True)
            elif target_pfa > np.max(pfa):
                init_thr -= torch.std(init_thr)
                # print(f'Target Pfa is higher than any Pfa, current pfa: {pfa}, current thresholds (*1e-3): {init_thr*1e3}', flush=True)
            else:
                #Check that the pfa is not e.g [1,0,0]
                if np.max(pfa) > target_pfa + delta_pfa:
                    init_thr = torch.linspace((init_thr[0] + init_thr[1]) / 2.0, init_thr[2], 3, device=device)
                    # print(f'Maximum Pfa > target+delta, current pfa: {pfa}, current thresholds (*1e-3): {init_thr*1e3}', flush=True)
                if np.min(pfa) < target_pfa - delta_pfa:
                    init_thr = torch.linspace(init_thr[0], (init_thr[1] + init_thr[2]) / 2.0, 3, device=device)
                    # print(f'Minimum Pfa < target-delta, current pfa: {pfa}, current thresholds (*1e-3): {init_thr*1e3}', flush=True)
    
    print(f'Final thresholds to yield Pfa={target_pfa} are: {init_thr}', flush=True)
    return init_thr


def testSensingFixedPfa(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                    theta_span_min_sens_test, theta_span_max_sens_test,
                    range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                    range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                    N, S, noiseVariance, delta_f, lambd, assumed_gain_phase, gain_phase, 
                    msg_card, refConst, threshold_list, batch_size, numTestSamples, 
                    target_pfa, delta_pfa,
                    device='cuda'):
    '''
    Function to test the sensing performance for a fixed false alarm probability
    probability.
    This function is very similar to testSensingROC, but here we just use the 
    threshold that gives us a false alarm probability close to the target value.
    For inputs and outputs, please check the testSensingROC function.
    '''
    print('**STARTED TESTING WITH FIXED PFA**', flush=True)
    with torch.no_grad():
        est_metric, est_angle, est_range, est_pos, \
        true_target, true_angle, true_range, true_pos \
             = computePositionsTesting(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                            theta_span_min_sens_test, theta_span_max_sens_test,
                            range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                            range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                            N, S, noiseVariance, delta_f, lambd, assumed_gain_phase, gain_phase, 
                            msg_card, refConst, batch_size, numTestSamples, device)
        
        #Get thresholds that give relatively close to the target Pfa
        init_thr = torch.clone(threshold_list)      #To avoid that the next function overwrites the thresholds.
        # print(f'The initial thresholds were: {init_thr}', flush=True)
        final_thr = obtainThresholdsFixedPfa(est_metric, true_target, target_pfa, delta_pfa, init_thr)
        # print(f'The final thresholds are: {final_thr}', flush=True)

        #Lists to save final results
        pd, pfa, mse_angle, mse_range, mse_pos = [], [], [], [], []
        #Compute detection and false alarm probabilities, and RMSEs
        for t in range(len(final_thr)):
            pd_temp, pfa_temp, mse_angle_temp, mse_range_temp, mse_pos_temp = \
                                getSensingMetrics(est_metric, true_target, 
                                                final_thr[t], est_angle, true_angle, est_range, true_range, est_pos, true_pos)
            pd.append(pd_temp)
            pfa.append(pfa_temp)
            mse_angle.append(mse_angle_temp)
            mse_range.append(mse_range_temp)
            mse_pos.append(mse_pos_temp)
        
        #Save data after target_pfa - delta_pfa < pfa < target_pfa + delta_pfa, but interpolation expects x-values to be ordered
        pfa_ordered = np.sort(pfa)
        idx_order = np.argsort(pfa)         #Save indices to keep the correspondence of pd-pfa values
        pd_inter = np.interp(-2,np.log10(pfa_ordered),np.array(pd)[idx_order])     
        pfa_inter = np.array(pfa).mean()
        mse_angle_inter = np.interp(-2,np.log10(pfa_ordered),np.array(mse_angle)[idx_order])
        mse_range_inter = np.interp(-2,np.log10(pfa_ordered),np.array(mse_range)[idx_order])
        mse_pos_inter = np.interp(-2,np.log10(pfa_ordered),np.array(mse_pos)[idx_order])

    return pd_inter, pfa_inter, mse_angle_inter, mse_range_inter, mse_pos_inter

def saveNetwork(save_path, network, optimizer):
    '''
    Function to save the state dictionary of a network and the optimizer.
    It print a message at the end.
    Inputs:
        - save_path: path where to save the network, including model name. String.
        - network: network to be saved. Instance of class torch.nn.Module.
        - optimizer: optimizer to be saved. Instance of class torch.optim.
    '''
    torch.save({
        'model': network.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, save_path)
    print('Model saved successfully', flush=True)
