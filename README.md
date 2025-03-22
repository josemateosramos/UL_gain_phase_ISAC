# UL_gain_phase_ISAC

## Getting Started
This code is based on Pytorch 1.12.1 and CUDA 11.3.1, and may not work with other versions. For more information about how to install these versions, check the Pytorch documentation.

The simulation parameters to train and test different scenarios are located in the simulation_parameters.py file within the lib/ directory. The methods/ directory contains the scripts to test and train (when applicable) the baseline and both unsupervised learning approaches. To run the code, only the files under the methods/ folder need to be run.

The inputs to the code are:

-g, --gainphase: Binary flag controlling whether we consider gain-phase impairments (0 for no impairments and 1 otherwise). Default value: 0.
-s, --seed: Integer controling the seed to use in the simulations. Default value: 10.
-l, --loss: Integer controling the loss function to use during training (0 for 'max' and 1 for 'reconstruction'). Default value: 0.

The script to obtain Fig. 2 in the paper attached below can be found under the plot_fig_2/ directory.

## Additional information
If you decide to use the source code for your research, please make sure to cite our paper:

J. M. Mateos-Ramos, C. Häger, M. F. Keskin, L. Le Magoarou, and H. Wymeersch, "Unsupervised Learning for Gain-Phase Impairment Calibration in ISAC Systems," in IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP), Hyderabad, India, 2025.
