# -*- coding: utf-8 -*-
from ..lib.simulation_parameters import *

#Test baseline 
pd_roc, pfa_roc, angle_mse_roc, range_mse_roc, pos_mse_roc \
    = testSensingROC(var_gain, theta_mean_min_sens_test, theta_mean_max_sens_test, 
                    theta_span_min_sens_test, theta_span_max_sens_test,
                    range_mean_min_sens_test, range_mean_max_sens_test, range_span_min_sens_test,
                    range_span_max_sens_test, Ngrid_angle, Ngrid_range,
                    N, S, noiseVariance, delta_f, lambd, assumed_gain_phase, gain_phase, 
                    msg_card, refConst, thresholds_roc, batch_size, nTestSamples, 
                    device=device)

#Save results with some information in the file name
file_name = 'simo_baseline'
if gain_imp_flag:
    file_name += '_gp_imp'
else:
    file_name += '_ideal'
file_name += '_realization_' + str(sim_seed)
np.savez('results/' + file_name, \
        pd = pd_roc, pfa = pfa_roc, angle_mse = angle_mse_roc, \
        range_mse = range_mse_roc, pos_mse = pos_mse_roc, \
        gain_phase = gain_phase)
