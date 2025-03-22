# -*- coding: utf-8 -*-
from ..lib.simulation_parameters import *

#Change seed for training
torch.manual_seed(64)
np.random.seed(64)
torch.cuda.manual_seed(64)

# Train network
mse_loss, num_iterations, pd_test, pfa_test, mse_angle_test, mse_range_test, mse_pos_test \
    = trainNetworkRXUL(network_impairment, optimizer, var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                    theta_span_min_sens_test, theta_span_max_sens_test,
                    range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                    range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                    N, S, noiseVariance, delta_f, lambd, gain_phase,
                    msg_card, refConst, thresholds_pfa, batch_size, train_it, test_iterations_list, 
                    nTestSamples, target_pfa, delta_pfa, sch_flag=scheduler_flag, scheduler=scheduler, 
                    loss=loss_training, device=device)

#Test performance
#Fix seed for reproducibility
torch.manual_seed(10)
np.random.seed(10)
torch.cuda.manual_seed(10)

pd_roc, pfa_roc, angle_mse_roc, range_mse_roc, pos_mse_roc \
    = testSensingROC(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                    theta_span_min_sens_test, theta_span_max_sens_test,
                    range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                    range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                    N, S, noiseVariance, delta_f, lambd, network_impairment.impairments, gain_phase, 
                    msg_card, refConst, thresholds_roc, batch_size, nTestSamples, 
                    device=device)

print('threholds: ', thresholds_roc, flush=True)
print('\npd: ', pd_roc, flush=True)
print('pfa: ', pfa_roc, flush=True)
print('\nangle_mse: ', angle_mse_roc, flush=True)
print('range_mse: ', range_mse_roc, flush=True)
print('pos_mse: ', pos_mse_roc, flush=True)

#Save results with some information in the file name
file_name = 'gp_learning_rx_ul' + loss_training
if gain_imp_flag:
    file_name += '_gp_imp'    
else:
    file_name += '_ideal'
file_name += '_lr_' + str(learning_rate)
if scheduler_flag:
    file_name += '_sch'
file_name += '_realization_' + str(sim_seed)

#Save learned model
saveNetwork('models/' + file_name + '_model', network_impairment, optimizer)

#Save metrics
np.savez('results/' + file_name, \
        loss = mse_loss, num_iterations = num_iterations, \
        pd = pd_roc, pfa = pfa_roc, angle_mse = angle_mse_roc, \
        range_mse = range_mse_roc, pos_mse = pos_mse_roc, \
        pd_test = pd_test, pfa_test = pfa_test, \
        mse_angle_test = mse_angle_test, mse_range_test = mse_range_test, \
        mse_pos_test = mse_pos_test, \
        learned_gp = network_impairment.impairments.detach().cpu(), \
        gain_phase = gain_phase)
