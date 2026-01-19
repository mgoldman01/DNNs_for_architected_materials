import torch
from collections import OrderedDict

def convert_state_dict(cp_path, convert_to='non-parallel'):
    """
    BLUF (or TL;DR for software folks): This function converts a model's weights for use with one or multiple GPUs

    PyTorch model weights are saved as .pth files and within python are structured as dictionaries, where each key is descriptive of the 
    respective layer of the model. For example, for a 2-layer neural network with torch.nn.Linear() layers, the first layer's weights would likely be:

    'model.linear0.weights' and 'model.linear0.bias'

    When a model is trained on multiple GPUs, the module torch.nn.DataParallel is used, which distributes training across the GPUs. The model parameters 
    then have ".module" added in, which is present in every layer's parameters' dictionary key name. Therefore, when loading a model for inference either
    on one GPU or multiple, the format of the weights must be considered and adjusted accordingly.
    (see https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)

    Therefore, this function adds or removes 'module.' from each key as directed.

    **NOTE: PyTorch refers to the dictionary of model weights as the "state_dict" and we have maintained that terminology here.**

    
    Args:
        cp_path (str): path to model checkpoint folder, in '.pth' format
        convert_to (str, optional): How to convert the model's weight dict. If the model was trained on more than one GPU and will be run on a single GPU, 
                                    convert to "non-parallel".
                                    If the model was trained on one GPU and will be run on multiple, convert to "parallel". 
                                    Defaults to 'non-parallel'.

    Raises:
        Exception: "convert_to must be either \'parallel\' or \'non-parallel\' "        i.e., get the keyword right

    Returns:
        OrderedDict: Weights dictionary of the model, modified for either one or many GPUs
    """
    
    state_dict = torch.load(cp_path)


    if convert_to == 'parallel':

        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k] = v
            else:
                name = 'module.'+k # remove `module.`
                new_state_dict[name] = v
        
    elif convert_to == 'non-parallel':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_key = k[7:]  # remove 'module.'
            else:
                new_key = k
            new_state_dict[new_key] = v
    else:
        raise Exception("convert_to must be either \'parallel\' or \'non-parallel\' ")
        
    return new_state_dict