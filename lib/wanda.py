import ssl
ssl._create_default_https_context=ssl._create_unverified_context
import torch 
import torch.nn as nn 
from .data import get_split_loader
from .layerwrapper import  WrappedCLIP
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

# 定义一个钩子函数
def all_data_hook_fn(module, input, output, layer_name):
    single_layer_outputs.setdefault(layer_name, []).append(output[0].detach().cpu())
    single_layer_inputs.setdefault(layer_name, []).append(input[0].detach().cpu())


def fast_register_hook(model):
    all_hooks=[]
    target=2
    for index,layer in enumerate(model.vision_model.encoder.layers):
        if index==target:
            break
        vname='visual'+str(index)
        vhandle=layer.register_forward_hook(lambda module, input, output, layer_name=vname: all_data_hook_fn(module, input, output, layer_name))
        all_hooks.append(vhandle)

    for index,layer in enumerate(model.text_model.encoder.layers):
        if index==target:
            break
        name='text'+str(index)
        handle=layer.register_forward_hook(lambda module, input, output, layer_name=name: all_data_hook_fn(module, input, output, layer_name))
        all_hooks.append(handle)
        
    return all_hooks


def prepare_calibration_input(model, args, dataloader, device):
    global single_layer_inputs
    global single_layer_outputs
    # all_hooks = register_hook(model)
    all_hooks = fast_register_hook(model)
    
    model.to(device)
    tokenizer = args.tokenizer
    attention_mask=None
    input_ids=None
    for batch in dataloader:
        images=batch['pixel_values']
        batch_size = images.shape[0]
        texts=batch["texts"]
        textinputs = tokenizer(texts, padding=True, return_tensors="pt")
        input_ids = textinputs['input_ids'].view(-1, textinputs['input_ids'].shape[-1])
        images=images.reshape(batch_size,3,224,224)
        attention_mask=textinputs['attention_mask']  # todo
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        # compute the embeddings
        if torch.cuda.is_available():
            for i in range(batch_size):
                image=images[i].unsqueeze(0)
                model.get_image_features(image.to(device))
                del image
                
                input_id=input_ids[i].unsqueeze(0)
                cur_mask=attention_mask[i].unsqueeze(0)
                model.get_text_features(input_ids=input_id,attention_mask=cur_mask)
                del input_id
                del cur_mask

        break

    del images
    del textinputs
    
    for item in single_layer_inputs:
        layer_inputs[item] = torch.cat(single_layer_inputs[item], dim=0)
        
    for item in single_layer_outputs:
        layer_outputs[item] = torch.cat(single_layer_outputs[item], dim=0)


    single_layer_inputs.clear()
    single_layer_outputs.clear()
    
    # model.to('cpu')
    vinps=layer_inputs['visual'+str(0)]
    inps=layer_inputs['text'+str(0)]
    vouts = torch.zeros_like(vinps)
    outs = torch.zeros_like(inps)
   
    layer_inputs.clear()
    
    for handle in all_hooks:
        handle.remove()
    
    torch.cuda.empty_cache()
    
    return vinps.detach(),vouts.detach(),inps.detach(),outs.detach(),attention_mask.detach()


def prune_layer(layer,args,prune_ins,prune_outs,attention_mask,causal_attention_mask,index_i,modal):
    # print(f"prune_outs:{prune_outs}")
    # print(f"prune_ins:{prune_ins}")
    res={}
    wrapped_layers = {}
    sparsity_ratio=0
    prune_n=args.prune_n
    prune_m=args.prune_m
    if modal=='visual':
        sparsity_ratio=args.imgsparsity_ratio
    else:
        sparsity_ratio=args.textsparsity_ratio
        
    subset = find_layers(layer)
    for name in subset:
        wrapped_layers[name] = WrappedCLIP(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            wrapped_layers[name].add_batch(inp[0], out[0])
        return tmp

    handles = []
    for name in wrapped_layers:
        handles.append(subset[name].register_forward_hook(add_batch(name)))    
        
    for j in range(args.nsamples):
        with torch.no_grad():
            if attention_mask!=None:
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=attention_mask[j],causal_attention_mask=causal_attention_mask)[0]
            else:
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=None,causal_attention_mask=None)[0]
                
    for h in handles:
        h.remove()
                
    for name in subset:
        # print(f"pruning layer {index_i} name {name}")
        curweight=subset[name].weight
        
        W_metric = torch.abs(curweight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))
        # W_metric = torch.abs(curweight.data) 
        
        W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
        if prune_n != 0:
            # structured n:m sparsity
            for ii in range(W_metric.shape[1]):
                if ii % prune_m == 0:
                    tmp = W_metric[:,ii:(ii+prune_m)].float()
                    W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
        else:    
            # sort_res = torch.sort(W_metric, dim=-1, stable=True)
            sorted_values, sorted_indices = torch.sort(W_metric, dim=-1, stable=True)
            
            # unstructured pruning
            indices = sorted_indices[:,:int(W_metric.shape[1]*sparsity_ratio)]
            W_mask.scatter_(1, indices, True)
    
        curweight.data[W_mask] = 0  ## set weights to zero 

    for j in range(args.nsamples):
        with torch.no_grad():
            if attention_mask!=None:
                # from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=attention_mask[j],causal_attention_mask=causal_attention_mask)[0]
            else:
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=None,causal_attention_mask=None)[0]
    
    # print(f"after prune_outs:{prune_outs}")


def prune_clip_with_wanda(args, model, device, prune_n=0, prune_m=0,preprocess=None):    
    #### Dataset #### 
    print("Creating retrieval dataset")
    # Get data loaders
    print("loading calibdation data")
    train_loader = get_split_loader('calibration', args.data_name, args.batch_size, args, preprocess)
    # val_loader = get_split_loader('test', args.data_name, args.batch_size, args.workers, args, preprocess)
    print("dataset loading complete")
    # print("Before pruning Starting validating...")
    
    visual_ins, visual_outs, text_ins, text_outs,attention_mask = prepare_calibration_input(model, args, train_loader, device)

    attention_mask = attention_mask.to(dtype=torch.bool,device=device)
    visual_ins.to(device)
    visual_outs.to(device)
    text_ins.to(device)
    text_outs.to(device)
    
    visual_ins.requires_grad = False
    visual_outs.requires_grad = False
    text_ins.requires_grad = False
    text_outs.requires_grad = False
    
    from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
    temp_s=torch.ones(1,text_ins.shape[1])
    causal_attention_mask = _create_4d_causal_attention_mask(
        temp_s.size(), text_ins.dtype, device=device
    )

    #visual encoder part:
    for index_i, layer in enumerate(model.vision_model.encoder.layers):
        layer.to(args.device)
        visual_ins=visual_ins.to(device)
        visual_outs=visual_outs.to(device)
        prune_layer(layer,args,visual_ins, visual_outs, None, None,index_i,'visual')
        visual_ins=visual_ins.to('cpu')
        visual_outs=visual_outs.to('cpu')
        visual_ins,visual_outs=visual_outs,visual_ins
        layer.to('cpu')
        
    
    # text encoder part
    for index_i, layer in enumerate(model.text_model.encoder.layers):
        layer.to(args.device)
        text_ins=text_ins.to(device)
        text_outs=text_outs.to(device)
        prune_layer(layer,args,text_ins, text_outs, attention_mask, causal_attention_mask,index_i,'text')
        text_ins=text_ins.to('cpu')
        text_outs=text_outs.to('cpu')
        text_ins,text_outs=text_outs,text_ins
        layer.to('cpu')

    torch.cuda.empty_cache()
    layer_inputs.clear()
    layer_outputs.clear() 
    return model