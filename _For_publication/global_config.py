import warnings
import torch
import numpy as np
import torch.optim as optim

class GlobalConfig(object):

    #*************Data Config*****************
    datapath = '_Data/Simulated_diffusion_tracks/'
    filename_X = '202111252237_simulated_diffusion_clean_changing_dim2_ntraces300000_D_randomFalse_dt0.03333333333333333s_LocErrRatio_1-16_R7-25_Len5-600_X.pkl'
    filename_y = '202111252237_simulated_diffusion_clean_changing_dim2_D_randomFalse_dt0.03333333333333333s_LocErrRatio_1-16_R7-25_Len5-600_timeresolved_y.pkl'
    
    the_data_is = '3D' if 'dim3' in filename_X else '2D'
    if the_data_is=='2D':
        #*************Standard Config*****************
        val_size = 0.2
        test_size = 0.2
        seed = 42
        seeds = [42] #[42, 99, 191, 12345, 0]
        X_padtoken = 0
        y_padtoken = 10
        shuffle = True

        #*************Training Config*****************
        lr = 3.9421*10**-5
        epochs = 100
        batch_size = 192
        optim_choice = optim.Adam

        #*************Model Config*****************
        features = ['XYZ', 'SL', 'DP']
        n_classes = 4 # number of classes to predict

        init_channels = 130
        channel_multiplier = 2

        pooling = 'max'
        pools = [2, 2, 2, 2, 2, 2, 2]

        depth = 4
        dil_rate = 2

        enc_conv_nlayers = 2
        dec_conv_nlayers = 1
        bottom_conv_nlayers = 3
        out_nlayers = 4

        kernelsize = 7
        outconv_kernel = 3

        batchnorm = True
        batchnormfirst = True
    
    if the_data_is=='3D':
        #*************Standard Config*****************
        val_size = 0.2
        test_size = 0.2
        seed = 42
        seeds = [42] #[42, 99, 191, 12345, 0]
        X_padtoken = 0
        y_padtoken = 10
        shuffle = True

        #*************Training Config*****************
        lr = 0.00020933097456506567
        epochs = 100
        batch_size = 256
        optim_choice = optim.RMSprop

        #*************Model Config*****************
        features = ['XYZ', 'SL', 'DP']
        n_classes = 4 # number of classes to predict

        init_channels = 48
        channel_multiplier = 2

        pooling = 'max'
        pools = [2, 2, 2, 2, 2, 2, 2]

        depth = 3
        dil_rate = 2

        enc_conv_nlayers = 3
        dec_conv_nlayers = 4
        bottom_conv_nlayers = 4
        out_nlayers = 2

        kernelsize = 5
        outconv_kernel = 3

        batchnorm = True
        batchnormfirst = True

    def _parse(self, kwargs):
        """
        update config based on kwargs dictionary
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        globals.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

globals = GlobalConfig()
