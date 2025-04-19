import ssl
ssl._create_default_https_context=ssl._create_unverified_context
import torch 
import torch.nn as nn 

layer_inputs = {}
layer_outputs= {}

global single_layer_inputs
global single_layer_outputs
single_layer_inputs = {}
single_layer_outputs= {}


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def magnitude_prune_layer(args,layer, prune_n=0, prune_m=0):
    subset = find_layers(layer)

    for name in subset:
        W = subset[name].weight.data 
        W_metric = torch.abs(W)
        if prune_n != 0:
            W_mask = (torch.zeros_like(W)==1)
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:
            thresh = torch.sort(W_metric.flatten().cpu())[0][int(W.numel()*args.sparsity_ratio)].cpu()
            W_mask = (W_metric<=thresh)

        W[W_mask] = 0

def prune_magnitude(args, layers, prune_n=0, prune_m=0):
    for i in range(len(layers)):
        magnitude_prune_layer(args,layers[i],prune_n,prune_m)
       
def prune_clip_with_magnitude(args,model):
    prune_magnitude(args,model.vision_model.encoder.layers,args.prune_n,args.prune_m)
    prune_magnitude(args,model.text_model.encoder.layers,args.prune_n,args.prune_m)

    return model
    