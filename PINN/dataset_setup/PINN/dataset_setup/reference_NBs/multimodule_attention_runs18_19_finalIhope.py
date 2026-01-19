import os
import workflow_utils_v3
import sys

from workflow_utils_v3.FileDirectory import Directory

dirs = Directory(rootpath = '/data/tigusa1/MLTO_UCAH/MLTO_2024/')

# Sets directory of entire package
# rootpath = '/data/tigusa1/MLTO_UCAH/MLTO_2023/'

nbpath = os.path.join(dirs._8_CNN3D_multiparam_regressor, 'CNN3D_paper_iterations/multi_MLP')
cp_dir = os.path.join(nbpath, 'model_CPs')

from torchsummary import summary

import pandas as pd
import numpy as np
import json
import glob
import os

# For plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import cycle
from plotly.colors import sequential, qualitative

from workflow_utils_v3.Dataset_Preprocessor import Dataset_Preprocessor as DataP
source_data_path = dirs.source_data_path

from workflow_utils.DatasetCreator import DatasetCreator as DC
source_data_path = dirs.source_data_path

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


import torch.utils.data as data
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
# import torchsummary

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from workflow_utils.CNN3D_Compressed_Spatial import ResidualBlock
from workflow_utils.Training_Utils import VoxelDataset, train, validate

params_run_dic = {'allphys': {'params': ['volFrac', 
                                        'CH_11 scaled', 'CH_22 scaled', 'CH_33 scaled', 'CH_44 scaled', 'CH_55 scaled', 'CH_66 scaled',
                                        'CH_12 scaled', 'CH_13 scaled','CH_23 scaled',
                                        'EH_11 scaled', 'EH_22 scaled', 'EH_33 scaled',
                                        'GH_23 scaled', 'GH_13 scaled', 'GH_12 scaled', 
                                        'vH_12 scaled', 'vH_13 scaled', 'vH_23 scaled', 'vH_21 scaled', 'vH_31 scaled','vH_32 scaled',
                                        'KH_11 scaled', 'KH_22 scaled', 'KH_33 scaled', 
                                        'kappaH_11 scaled', 'kappaH_22 scaled', 'kappaH_33 scaled'],

        'params_segmented' : [['volFrac',], 
                             ['CH_11 scaled', 'CH_22 scaled', 'CH_33 scaled', 'CH_44 scaled', 'CH_55 scaled', 'CH_66 scaled',],
                             ['CH_12 scaled', 'CH_13 scaled','CH_23 scaled',],
                             ['EH_11 scaled', 'EH_22 scaled', 'EH_33 scaled',],
                             ['GH_23 scaled', 'GH_13 scaled', 'GH_12 scaled',],
                             ['vH_12 scaled', 'vH_13 scaled', 'vH_23 scaled', 'vH_21 scaled', 'vH_31 scaled','vH_32 scaled',],
                             ['KH_11 scaled', 'KH_22 scaled', 'KH_33 scaled',],
                             ['kappaH_11 scaled', 'kappaH_22 scaled', 'kappaH_33 scaled']],
                    },

        'allphys_unscaled': {'params': ['volFrac', 
                                        'CH_11', 'CH_22', 'CH_33', 'CH_44', 'CH_55', 'CH_66',
                                        'CH_12', 'CH_13','CH_23',
                                        'EH_11', 'EH_22', 'EH_33',
                                        'GH_23', 'GH_13', 'GH_12', 
                                        'vH_12', 'vH_13', 'vH_23', 'vH_21', 'vH_31','vH_32',
                                        'KH_11', 'KH_22', 'KH_33', 
                                        'kappaH_11', 'kappaH_22', 'kappaH_33'], 

        'params_segmented' : [['volFrac',], 
                             ['CH_11', 'CH_22', 'CH_33', 'CH_44', 'CH_55', 'CH_66',],
                             ['CH_12', 'CH_13','CH_23',],
                             ['EH_11', 'EH_22', 'EH_33',],
                             ['GH_23', 'GH_13', 'GH_12',],
                             ['vH_12', 'vH_13', 'vH_23', 'vH_21', 'vH_31','vH_32',],
                             ['KH_11', 'KH_22', 'KH_33',],
                             ['kappaH_11', 'kappaH_22', 'kappaH_33']],
                    },
             }

run_model_dic = {
    'run18': 'v4_multimod_attn_8mlp_wRelu_longer_featvec.py',
    'run19': 'v10_multimod_attn_8mlp_wRelu_long_featvec_full_batchnorm.py',
}


# Prelims for DatasetCreator
source_data_path = dirs.source_data_path


csvname = 'data_3phys_w_dyn_topos_scaled_PNs.csv'

batch_size= 32


from workflow_utils.Training_Utils import train, validate
from workflow_utils_v3.VoxelDataset import VoxelDataset


import os
import importlib.util

model_directory = os.path.join(nbpath, 'multimod_models')
models = sorted([file for file in os.listdir(model_directory) if file.endswith('.py') and file.startswith('v')])

testset_seed = seed = 17
volfrac_range = (0.01, 0.98)
datapre = DataP(csv_fn = csvname, volfrac_range = volfrac_range)
datapre.TrainTestSplit(test_set_topo_counts=[2,2,2], translate=True, testset_seed=seed)
print(datapre.idxTr.shape, datapre.idxVal.shape, datapre.idxTe.shape)
print(datapre.testsubset)

EPOCHS = 125

for Y_file_suffix, config_dic in params_run_dic.items():
    
    params = config_dic['params']
    params_segmented = config_dic['params_segmented']

    for run, model in run_model_dic.items():


        model_file_name = model
        print(model_file_name)
        modeltitle = model_file_name[:-3]
        ver = model_file_name.split('_')[0]
        # params = config['params']
        num_params = len(params)
        # params_segmented = config['params_segmented']
        num_modules = len(params_segmented)
        # Y_file_suffix = config['Y_file_suffix']


        fname_base = f'{run}_sd{seed}_augm_vf0p98_{modeltitle}_{num_params}par_{Y_file_suffix}'

        # Create the Datasets and DataLoaders
        trdat = VoxelDataset(datapre.idxTr, dirs.voxel_PN_filepath, params)
        trloader = DataLoader(trdat, batch_size = batch_size, shuffle=True)

        valdat = VoxelDataset(datapre.idxVal, dirs.voxel_PN_filepath, params)
        valloader = DataLoader(valdat, batch_size = batch_size, shuffle=True)

        tedat = VoxelDataset(datapre.idxTe, dirs.voxel_PN_filepath, params)
        teloader = DataLoader(tedat, batch_size = batch_size, shuffle=False)


        # Import the model from its respective module
        # model_file_name is defined above

        model_class_name = f'CNN3D_multimod_attn_{ver}'
        model_path = os.path.join(model_directory, model_file_name)
        # print(model_path)

        spec = importlib.util.spec_from_file_location(model_class_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Now you can access your model class
        CNN3D = getattr(module, model_class_name)
        cnn = torch.nn.DataParallel(CNN3D(params_segmented).to(device))

#         if num == 0:
#             # Generate and save a summary
#             summary_path = os.path.join(nbpath, 'model_summaries', f'{fname_base}_model_summary.txt')

#             with open(summary_path, 'w') as f:
#                 # Redirect stdout to the file
#                 sys.stdout = f

#                 # Generate the summary
#                 summary(cnn.eval(), input_size=(1,64,64,64)) 

#                 # Reset stdout to its default value (the terminal)
#                 sys.stdout = sys.__stdout__

#             f.close()
#         else:
#             pass
        cnn.train()

        lossfunc = torch.nn.L1Loss() # this is MAE loss
        lossfunc_name = 'MAE'
        optimizer = optim.Adam(cnn.parameters(), lr=0.001)

        cp_dir = os.path.join(nbpath, 'model_CPs')

        # This is for loading the model weights for continued training (which will be the case as of 7.25)
        numparams = len(params)

        cp_name = f'CP_{fname_base}.pth'
        best_weights_path = os.path.join(cp_dir, cp_name)
        print(best_weights_path)

        patience = 75

        min_val_loss = float('inf')
        best_val_loss = float('inf')
        early_stop_counter = 0
        # earlystop_min_delta = 0.000075
        earlystop_min_delta = 0.00075 # For L1Loss (MAE)

        # os.makedirs(best_weights_path, exist_ok=True)
        best_epoch = 0

        train_losses = []
        val_losses = []

        epochs_completed=0

        try:

            for epoch in range(EPOCHS):
                # Train the model
                train_loss = train(cnn, trloader, lossfunc, optimizer)

                # Validate the model
                val_loss = validate(cnn, valloader, lossfunc)


                # Save the model's weights if validation loss is improved
                improvement_delta = best_val_loss - val_loss

                if val_loss < best_val_loss:
                    pct_improved = (best_val_loss - val_loss) / best_val_loss * 100
                    print(f"Val loss improved from {best_val_loss:.5f} to {val_loss:.5f} ({pct_improved:.2f}% improvement) saving model state...")
                    best_val_loss = val_loss
                    torch.save(cnn.state_dict(), best_weights_path)  # Save model weights to file
                else:
                    print(f'Val loss did not improve from {best_val_loss:.5f}.')
                    # early_stop_counter += 1  # Increment early stopping counter

                if improvement_delta > earlystop_min_delta:
                    early_stop_counter = 0
                else:
                    early_stop_counter +=1


                # Collect model training history
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                # Check for early stopping
                if early_stop_counter >= patience:
                    print(f'Validation loss did not improve for {early_stop_counter} epochs. Early stopping...')
                    cnn.load_state_dict(torch.load(best_weights_path))
                    print(f"Model best weights restored - training epoch {best_epoch}")
                    break

                print(f'Epoch [{epoch+1}/{EPOCHS}]\tTrain Loss: {train_loss:.5f}\tValidation Loss: {val_loss:.5f}')

                epochs_completed +=1


            # Load the best weights at end of training epochs
            cnn.load_state_dict(torch.load(best_weights_path))  # Load best model weights
            print(f'Training epochs completed, best model weights restored - epoch {best_epoch}')
            min_val_loss = best_val_loss

        except KeyboardInterrupt:
            hist_dict = {f'train_loss {lossfunc_name}': train_losses, f'val_loss {lossfunc_name}': val_losses}
            cnn.load_state_dict(torch.load(best_weights_path))

        hist_dict = {f'train_loss {lossfunc_name}': train_losses, f'val_loss {lossfunc_name}': val_losses}

        histno=1
        histpath = os.path.join(nbpath,'model_jsons',f'{cp_name[:-4]}_training_history{histno}_{epochs_completed}ep.json')

        cnn.eval()

        with open(histpath, 'w') as f:
            json.dump(hist_dict, f)

        idxcols = ['topology_family', 'cell_type', 'dim_idx',] # 'volFrac', -- removed because this run includes it as a target predicted parameter
        idxcols.extend(params)

        predcols = [f'Predicted {par}' for par in params]

        # test_split_dataframe = dataobj.splitdfs['idxTe'][idxcols]
        test_split_dataframe = datapre.idxTe[idxcols]

        """
        Runs each Voxel array in the Testing dataset through the trained CNN, produces a dataframe
        of the predictions under the columns of "pred[icted] [parameter name]"
        """
        predsarray_batched = []

        for batch in teloader:
            arrays, param_vecs = batch

            with torch.no_grad():
                outputs = cnn(arrays)

            predsarray_batched.append(outputs.cpu().numpy())

        predsarray = np.concatenate(predsarray_batched, axis=0)

        predictions_dataframe = pd.DataFrame(predsarray, columns=predcols)

        test_split_dataframe = test_split_dataframe.join(predictions_dataframe)


        for col in predcols:
            par = col[10:]
            rescol = f'Residual {par}'
            test_split_dataframe[rescol] = test_split_dataframe[par] - test_split_dataframe[col]

            pctcol = f'Pct error {par[:-7]}'

            test_split_dataframe[pctcol] = (test_split_dataframe[col] - test_split_dataframe[par]) / test_split_dataframe[par] *100


        MAE = True

        test_error = 0
        for col in predcols:
            par = col[10:]
            if MAE: # Calculates Mean Absolute Error
                test_error +=  test_split_dataframe[f'Residual {par}'].abs().sum() # take sum of absolute error

            else: # calculates Root Mean Squared Error
                test_error += (test_split_dataframe[f'Residual {par}']**2).sum()**0.5

        test_error /= len(test_split_dataframe) # Take mean

        # print(f"Error of Test set predictions: {test_error:.4f}")

        totalerrors = []
        for col in predcols:
            par = col[10:]
            test_error = 0
            if MAE: # Calculates Mean Absolute Error
                test_error +=  test_split_dataframe[f'Residual {par}'].abs().sum() # take sum of absolute error

            else: # calculates Root Mean Squared Error
                test_error += (test_split_dataframe[f'Residual {par}']**2).sum()**0.5

            test_error /= len(test_split_dataframe) # Take mean

            if par == 'volFrac':
                print(f'Error of {par}: \t{test_error:.4f}')
            else:
                print(f'Error of {par[:-7]}: \t{test_error:.4f}')
            totalerrors.append(test_error)

        totalerrors.insert(0, 'mean')


        errorcols = ['cell_type']
        errorcols.extend([i[10:] for i in predcols])
        errordf = pd.DataFrame(columns = errorcols)
        errordf['cell_type'] = test_split_dataframe.cell_type.unique()



        for col in predcols:

            par = col[10:]

            errorseries = []

            for celltype in test_split_dataframe.cell_type.unique():

                celltypedf = test_split_dataframe[test_split_dataframe.cell_type == celltype]
                test_error = 0
                if MAE: # Calculates Mean Absolute Error
                    test_error +=  celltypedf[f'Residual {par}'].abs().sum() # take sum of absolute error

                else: # calculates Root Mean Squared Error
                    test_error += (celltypedf[f'Residual {par}']**2).sum()**0.5

                test_error /= len(celltypedf) # Take mean

                errorseries.append(test_error)

            errordf[par] = np.asarray(errorseries)

        errordf.loc[len(errordf.index)] = totalerrors

        cols = errordf.columns[2:]
        means = []
        for i in errordf.index:
            celltype_mean = errordf.loc[i][cols].mean()
            means.append(celltype_mean)
        errordf['cell_type mean error'] = means

        csvpath = os.path.join(nbpath, 'preds_and_plots', f'{fname_base}_test_set_errors.csv')
        errordf.to_csv(csvpath)


        """
        This cell produces plots of each Predicted parameter versus the actual value, as well as either residuals or percentage error.
        Plots are:
        (1) Predicted vs. actual parameter
        (2) volume fraction vs residual or percentage error
        """

        ###################This section contains parameters for defining particulars of the plots ##################

        # As shipped, *second_plot* can be of either residuals or percentage error 
        second_plot = 'Residual'# or 'Pct error' 

        width = 1400  # width of plot
        height_incr = 450 # height of each subplot

        # color plot markers by individual cell type or topology family
        category =  'cell_type'# or 'topology_family'

        # Name of plot to save
        img_name = f'{fname_base}_Multiparam{numparams}_prediction_plots'
        img_path = os.path.join(nbpath, 'preds_and_plots', img_name)

        # if desired to save the plot as a PNG, set to True
        save_img = True

        ##########################

        colors = cycle(qualitative.G10)


        predsdf = test_split_dataframe

        # colnames = ['Predicted {param}'.format(param=param), f'{second_plot}']
        # rownames = params

        # Makes subplots equivalent to the number of parameters Predicted
        fig = make_subplots(rows=len(params), cols=2, row_heights=[20 for i in range(len(params))], column_widths=[15,15],
                            column_titles = ['Predicted vs Actual', f'{second_plot}'.capitalize()],
                            row_titles = params)

        # Iterates over rows
        row=1
        for param in params:
            # colors and colors2 definitions ensure that the predicted vs. actual and residuals plot the same colors with the same chosen category
            colors = cycle(qualitative.G10)
            colors2 = cycle(qualitative.G10)

            # Plots predictions by category as selected above
            for uc in predsdf[f'{category}'].unique():

                df = predsdf[predsdf[f'{category}']==uc]

                fig.append_trace(go.Scatter(x=df[f'Predicted {param}'], 
                                            y=df[f'{param}'], 
                                            mode='markers', marker=go.scatter.Marker(color=next(colors)), legendgroup='1',
                                            showlegend=True, name=uc), row=row, col=1, )


            # Plots second_plot by category, both as selected above
            for uc in predsdf[f'{category}'].unique():

                df = predsdf[predsdf[f'{category}']==uc]

                # df = predsdf[predsdf.cell_type==uc]
                fig.append_trace(go.Scatter(x=df['volFrac'], 
                                            y=df[f'{second_plot} {param}'],  
                                            #y=df[df[f'{second_plot} {param}'].between(-100,100)], # This can filter out extremely large percentage errors if desired
                                            mode='markers', marker=go.scatter.Marker(color=next(colors2)), legendgroup='2',
                                            showlegend=True, name=uc), row=row, col=2, )

            # Plots a line of X=Y on all Predicted-vs-actual plots
            x = np.linspace(predsdf['Predicted {param}'.format(param=param)].min(), predsdf['Predicted {param}'.format(param=param)].max(),100)
            y = x
            fig.add_trace(go.Scatter(x=x, y=y, name='Predicted = actual', legendgroup='1',marker=go.scatter.Marker(color=next(colors))), row=row, col=1 )

            # Updates figure with labels
            fig.update_xaxes(title_text="Predicted {param}".format(param=param), row=row, col=1)
            fig.update_yaxes(title_text="Actual {param}".format(param=param), row=row, col=1)
            fig.update_xaxes(title_text="Volume Fraction", row=row, col=2)
            fig.update_yaxes(title_text=f"{second_plot.capitalize()}", row=row, col=2)
            fig.update_layout(title_text = f'Predictions & {second_plot.capitalize()}</br>')

            # fig = go.Figure(data=go.Scatter(x=x, y=x**2))

            fig.update_xaxes(griddash='solid', minor_griddash="solid")
            fig.update_yaxes(griddash='solid', minor_griddash="solid")
            row+=1

        fig.update_layout(
            title={
                'text': f'Plots of test data - multi-parameter prediction', #<br>{runnum} </br>
                'xref':"paper",
                'xanchor':'center',
                'x':0.5},
                height=height_incr*len(params), width=width,
                legend_tracegroupgap = 50*len(params))
        # fig.show()

        if save_img:
            # img_path = os.path.join(nbpath, img_name)
            fig.write_image(f'{img_path}.png')
        else:
            pass

        cnn.cpu()
        del cnn
            
            
# 'run0': {'model': 'v0_multimod_attn_4_mods.py',
# 'params': ['volFrac', 'CH_11 scaled', 'CH_12 scaled','CH_44 scaled', 'EH_11 scaled', 'GH_23 scaled',
#                     'kappaH_11 scaled', 'kappaH_22 scaled', 'vH_12 scaled', 'vH_23 scaled', 
#                     'KH_11 scaled', 'KH_22 scaled'], 

# 'params_segmented' : [['volFrac', 'CH_11 scaled', 'CH_12 scaled','CH_44 scaled', 'EH_11 scaled', 'GH_23 scaled',],
#                     ['kappaH_11 scaled', 'kappaH_22 scaled'], ['vH_12 scaled', 'vH_23 scaled'], 
#                     ['KH_11 scaled', 'KH_22 scaled']],
# 'Y_file_suffix': 'vfC3EGv2K2kap2'
#                     },
       
       
# 'run1': {'model': 'v1_multimod_attn_8_mods_smaller.py',
# 'params': ['volFrac', 'CH_11 scaled', 'CH_12 scaled', 'CH_44 scaled', 'GH_23 scaled','kappaH_11 scaled', 'vH_12 scaled',
#                     'KH_11 scaled'], 

# 'params_segmented' : [['volFrac',], ['CH_11 scaled',], ['CH_12 scaled',], ['CH_44 scaled',], ['GH_23 scaled',],
#                     ['kappaH_11 scaled',], ['vH_12 scaled',], ['KH_11 scaled',]],
                    
# 'Y_file_suffix': 'vfC2GkapvK'
#                     },
                    
# 'run2': {'model': 'v1_multimod_attn_8_mods_smaller.py',
# 'params': ['volFrac', 'vH_13 scaled', 'vH_23 scaled',  'vH_31 scaled',
#             'vH_32 scaled', 'KH_11 scaled', 'KH_22 scaled', 'kappaH_11 scaled'], 

# 'params_segmented' : [['volFrac',], ['vH_13 scaled',], ['vH_23 scaled',],  ['vH_31 scaled',],
#             ['vH_32 scaled',], ['KH_11 scaled',], ['KH_22 scaled',], ['kappaH_11 scaled',]],
                    
# 'Y_file_suffix': 'vfv4K2kap1'
#                     },
                    
# 'run3': {'model': 'v2_multimod_attn_6_mods.py',
#         'params': ['volFrac', 'CH_11 scaled', 'CH_22 scaled', 'CH_33 scaled', 'CH_44 scaled', 'CH_55 scaled', 'CH_66 scaled',
#                      'CH_12 scaled', 'CH_13 scaled','CH_23 scaled',
#                      'EH_11 scaled', 'EH_22 scaled', 'EH_33 scaled',
#                      'GH_23 scaled', 'GH_13 scaled', 'GH_12 scaled', 
#                      'vH_12 scaled', 'vH_13 scaled', 'vH_23 scaled', 'vH_21 scaled', 'vH_31 scaled','vH_32 scaled',], 

#         'params_segmented' : [['volFrac',], ['CH_11 scaled', 'CH_22 scaled', 'CH_33 scaled', 'CH_44 scaled', 'CH_55 scaled', 'CH_66 scaled',],
#                              ['CH_12 scaled', 'CH_13 scaled','CH_23 scaled',],
#                              ['EH_11 scaled', 'EH_22 scaled', 'EH_33 scaled',],
#                              ['GH_23 scaled', 'GH_13 scaled', 'GH_12 scaled',],
#                              ['vH_12 scaled', 'vH_13 scaled', 'vH_23 scaled', 'vH_21 scaled', 'vH_31 scaled','vH_32 scaled',]] ,

#         'Y_file_suffix': 'vfCmtxEGv'
#                     },
                    
# 'run4': {'model': 'v3_multimod_attn_8_mods_bigger.py',
#         'params': ['volFrac', 
#                     'CH_11 scaled', 'CH_22 scaled', 'CH_33 scaled', 'CH_44 scaled', 'CH_55 scaled', 'CH_66 scaled',
#                     'CH_12 scaled', 'CH_13 scaled','CH_23 scaled',
#                     'EH_11 scaled', 'EH_22 scaled', 'EH_33 scaled',
#                     'GH_23 scaled', 'GH_13 scaled', 'GH_12 scaled', 
#                     'vH_12 scaled', 'vH_13 scaled', 'vH_23 scaled', 'vH_21 scaled', 'vH_31 scaled','vH_32 scaled',
#                     'KH_11 scaled', 'KH_22 scaled', 'KH_33 scaled', 
#                     'kappaH_11 scaled', 'kappaH_22 scaled', 'kappaH_33 scaled'], 

#         'params_segmented' : [['volFrac',], 
#                              ['CH_11 scaled', 'CH_22 scaled', 'CH_33 scaled', 'CH_44 scaled', 'CH_55 scaled', 'CH_66 scaled',],
#                              ['CH_12 scaled', 'CH_13 scaled','CH_23 scaled',],
#                              ['EH_11 scaled', 'EH_22 scaled', 'EH_33 scaled',],
#                              ['GH_23 scaled', 'GH_13 scaled', 'GH_12 scaled',],
#                              ['vH_12 scaled', 'vH_13 scaled', 'vH_23 scaled', 'vH_21 scaled', 'vH_31 scaled','vH_32 scaled',],
#                              ['KH_11 scaled', 'KH_22 scaled', 'KH_33 scaled',],
#                              ['kappaH_11 scaled', 'kappaH_22 scaled', 'kappaH_33 scaled']],

#         'Y_file_suffix': 'allphys'
#                     },
    
# 'run5': {'model': 'v1_multimod_attn_8_mods_smaller.py',
#         'params': ['volFrac', 'vH_13', 'vH_23',  'vH_31',
#                     'vH_32', 'KH_11', 'KH_22', 'kappaH_11'], 

#         'params_segmented' : [['volFrac',], ['vH_13',], ['vH_23',],  ['vH_31',],
#                     ['vH_32',], ['KH_11',], ['KH_22',], ['kappaH_11',]],

#         'Y_file_suffix': 'vfv4K2kap1_unscaled'
#                             },
                    
# 'run6': {'model': 'v2_multimod_attn_6_mods.py',
#         'params': ['volFrac', 'CH_11', 'CH_22', 'CH_33', 'CH_44', 'CH_55', 'CH_66',
#                      'CH_12', 'CH_13','CH_23',
#                      'EH_11', 'EH_22', 'EH_33',
#                      'GH_23', 'GH_13', 'GH_12', 
#                      'vH_12', 'vH_13', 'vH_23', 'vH_21', 'vH_31','vH_32',], 

#         'params_segmented' : [['volFrac',], ['CH_11', 'CH_22', 'CH_33', 'CH_44', 'CH_55', 'CH_66',],
#                              ['CH_12', 'CH_13','CH_23',],
#                              ['EH_11', 'EH_22', 'EH_33',],
#                              ['GH_23', 'GH_13', 'GH_12',],
#                              ['vH_12', 'vH_13', 'vH_23', 'vH_21', 'vH_31','vH_32',]] ,

#         'Y_file_suffix': 'vfCmtxEGv_unscaled'
#                             },

