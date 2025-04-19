import ssl
ssl._create_default_https_context=ssl._create_unverified_context
import torch 
import torch.nn as nn 
from .sparsegpt import  SparseCLIP
from .data import get_split_loader
import numpy as np
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
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

def check_sparsity(model):
    vlayers = model.vision_model.encoder.layers
    layers = model.text_model.encoder.layers
    
    count = 0 
    total_params = 0
    
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"text layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    for i in range(len(vlayers)):
        layer = vlayers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"visual layer {i} sparsity {float(sub_count)/sub_params:.6f}")
        
    return float(count)/total_params 

    
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


def prune_clip_with_sliding_sparsegpt(args, model, device, prune_n=0, prune_m=0,preprocess=None):
    #### Dataset #### 
    print("Creating retrieval dataset")
    # Get data loaders
    print("loading calibdation data")
    train_loader = get_split_loader('calibration', args.data_name, args.split_batch_size, args)
    print("dataset loading complete")

    visual_ins, visual_outs, text_ins, text_outs,attention_mask = prepare_calibration_input(model, args, train_loader, device)
    
    attention_mask = attention_mask.to(dtype=torch.bool,device=device)
        
    visual_ins.requires_grad = False
    visual_outs.requires_grad = False
    text_ins.requires_grad = False
    text_outs.requires_grad = False
    
    temp_s=torch.ones(1,text_ins.shape[1])
    # from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
    causal_attention_mask = _create_4d_causal_attention_mask(
        temp_s.size(), text_ins.dtype, device=device
    )
    
    for name,param in model.named_parameters():
        param.requires_grad = False
    
    if args.only_visual!=1:
        text_ins=text_ins.to(device)
        text_outs=text_outs.to(device)
        #text encoder part
        sliding_process(model.text_model.encoder.layers,text_ins,text_outs,args,device,'text',args.train_epoch,1e-5,attention_mask,causal_attention_mask)
        del text_ins
        del text_outs
    
    visual_ins=visual_ins.to(device)
    visual_outs=visual_outs.to(device)
    #visual encoder part:
    sliding_process(model.vision_model.encoder.layers,visual_ins,visual_outs,args,device,'visual',args.train_epoch,1e-5)
    del visual_ins
    del visual_outs
    
    
    layer_inputs.clear()
    layer_outputs.clear() 
    torch.cuda.empty_cache()   
    return model


def sliding_process(sliding_layers,sliding_ins,sliding_outs,args,device,modal,max_epochs,lr,attention_mask=None,causal_attention_mask=None):
    sliding_length=len(sliding_layers)
    index_i=0
    cur_next_index=index_i+1
    for item in sliding_layers:
        item.requires_grad = False
    
    cur_normal_output=layer_outputs[modal+str(index_i)].detach()
    cur_normal_output=cur_normal_output.to(device)
    next_normal_output=torch.zeros_like(sliding_outs,device=device)
 
    while(index_i<sliding_length):
        cur_lr=lr
        if index_i!=sliding_length-1:
            cur_layer=sliding_layers[index_i]
            next_layer=sliding_layers[cur_next_index]
            with torch.no_grad():
                for j in range(args.nsamples):
                    if attention_mask!=None:
                        next_normal_output[j] = next_layer(cur_normal_output[j].unsqueeze(0), attention_mask=attention_mask[j],causal_attention_mask=causal_attention_mask)[0]
                    else:
                        next_normal_output[j] = next_layer(cur_normal_output[j].unsqueeze(0), attention_mask=None,causal_attention_mask=None)[0]    
    
            sparsegpt_prune_layer(cur_layer,args,sliding_ins, sliding_outs, attention_mask, causal_attention_mask,index_i,modal)
            
            cur_layer_output=sliding_outs.detach()

            criterion = nn.MSELoss() 

            next_layer.requires_grad = True    
            for name,param in next_layer.named_parameters():
                if 'weight' in name:
                    param.requires_grad=True
            optimizer = torch.optim.AdamW(next_layer.parameters(), lr=cur_lr)
            next_layer.train()
            
            # print(f"dropout:{next_layer.self_attn.dropout}")
            for epoch in range(max_epochs):
                epoch_loss=1
                for j in range(args.nsamples):
                    optimizer.zero_grad()
                    # 前向传播
                    if attention_mask!=None:
                        output = next_layer(cur_layer_output[j].unsqueeze(0), attention_mask=attention_mask[j],causal_attention_mask=causal_attention_mask)[0]
                    else:
                        output = next_layer(cur_layer_output[j].unsqueeze(0), attention_mask=None,causal_attention_mask=None)[0]
                    # 计算损失
                    loss = criterion(output, next_normal_output[j].unsqueeze(0))
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    epoch_loss+=loss
                    # print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
                    
                epoch_loss=epoch_loss/args.nsamples
                # print(f"epoch {epoch} aver loss:{epoch_loss}")
                if(epoch_loss<args.train_epoch_loss_threshold):
                    break
                    
            
            next_layer.requires_grad = False
            for name,param in next_layer.named_parameters():
                if 'weight' in name:
                    param.requires_grad=False
            next_layer.eval()
            
            sliding_ins,sliding_outs=sliding_outs,sliding_ins
            
        else: #means pruning at the last block ,prune without any training
            cur_layer=sliding_layers[index_i]
            sparsegpt_prune_layer(cur_layer,args,sliding_ins, sliding_outs, attention_mask, causal_attention_mask,index_i,modal)
            
        cur_normal_output,next_normal_output =next_normal_output,cur_normal_output
         
        index_i+=1
        cur_next_index+=1
    
    del cur_normal_output
    del next_normal_output
    torch.cuda.empty_cache()


@torch.no_grad()
def sparsegpt_prune_layer(layer,args,prune_ins,prune_outs,attention_mask,causal_attention_mask,index_i,modal):
    # print(f"prune_outs:{prune_outs}")
    # print(f"prune_ins:{prune_ins}")
    if modal=='visual':
        sparsity_ratio=args.imgsparsity_ratio
    elif modal=='text':
        sparsity_ratio=args.textsparsity_ratio
    
    subset = find_layers(layer)

    gpts = {}
    for name in subset:
        gpts[name] = SparseCLIP(subset[name])

    def add_batch(name):
        def tmp(_, inp, out):
            gpts[name].add_batch(inp[0], out[0])
        return tmp

    handles = []
    for name in gpts:
        handles.append(subset[name].register_forward_hook(add_batch(name)))


    for j in range(args.nsamples):
        with torch.no_grad():
            if attention_mask!=None:
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=attention_mask[j],causal_attention_mask=causal_attention_mask)[0]
            else:
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=None,causal_attention_mask=None)[0]
    for h in handles:
        h.remove()

    for name in gpts:
        # print(index_i, name)
        # print('Pruning ...')

        gpts[name].fasterprune(sparsity_ratio, percdamp=0.01, blocksize=args.nsamples,prune_n=args.prune_n,prune_m=args.prune_m)
        gpts[name].free()

    for j in range(args.nsamples):
        with torch.no_grad():
            if attention_mask!=None:
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=attention_mask[j],causal_attention_mask=causal_attention_mask)[0]
            else:
                prune_outs[j] = layer(prune_ins[j].unsqueeze(0), attention_mask=None,causal_attention_mask=None)[0]
        
    # print(f"after prune_outs:{prune_outs}")
  

def second_prune_clip_with_sliding_sparsegpt(args, model, device, prune_n=0, prune_m=0,preprocess=None):
    #### Dataset #### 
    print("Creating retrieval dataset")
    # Get data loaders
    print("loading calibdation data")
    train_loader = get_split_loader('calibration', args.data_name, args.split_batch_size, args)
    print("dataset loading complete")

    visual_ins, visual_outs, text_ins, text_outs,attention_mask = prepare_calibration_input(model, args, train_loader, device)
    
    attention_mask = attention_mask.to(dtype=torch.bool,device=device)
        
    visual_ins.requires_grad = False
    visual_outs.requires_grad = False
    text_ins.requires_grad = False
    text_outs.requires_grad = False
    
    temp_s=torch.ones(1,text_ins.shape[1])
    # from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
    causal_attention_mask = _create_4d_causal_attention_mask(
        temp_s.size(), text_ins.dtype, device=device
    )
    
    for name,param in model.named_parameters():
        param.requires_grad = False
    
    if not args.only_visual:
        text_ins=text_ins.to(device)
        text_outs=text_outs.to(device)
        #text encoder part
        sliding_process(model.text_model.encoder.layers,text_ins.clone(),text_outs.clone(),args,device,'text',args.train_epoch,1e-5,attention_mask,causal_attention_mask)
        args.textsparsity_ratio=args.step2_prune_ratio
        sliding_process(model.text_model.encoder.layers,text_ins,text_outs,args,device,'text',args.train_epoch,1e-5,attention_mask,causal_attention_mask)
        del text_ins
        del text_outs
    
    visual_ins=visual_ins.to(device)
    visual_outs=visual_outs.to(device)
    #visual encoder part:
    sliding_process(model.vision_model.encoder.layers,visual_ins.clone(),visual_outs.clone(),args,device,'visual',args.train_epoch,1e-5)
    args.imgsparsity_ratio=args.step2_prune_ratio
    sliding_process(model.vision_model.encoder.layers,visual_ins,visual_outs,args,device,'visual',args.train_epoch,1e-5)
    del visual_ins
    del visual_outs
    
    
    layer_inputs.clear()
    layer_outputs.clear() 
    torch.cuda.empty_cache()   
    return model
 
  

def prune_clip_with_sparsegpt(args, model, device, prune_n=0, prune_m=0,preprocess=None):
    #### Dataset #### 
    print("Creating retrieval dataset")
    # Get data loaders

    print("loading calibdation data")
    train_loader = get_split_loader('calibration', args.data_name, args.batch_size, args)
    print("dataset loading complete")

    visual_ins, visual_outs, text_ins, text_outs,attention_mask = prepare_calibration_input(model, args, train_loader, device)

    attention_mask = attention_mask.to(dtype=torch.bool,device=device)
    visual_ins=visual_ins.to(device)
    visual_outs=visual_outs.to(device)
    text_ins=text_ins.to(device)
    text_outs=text_outs.to(device)
    
    visual_ins.requires_grad = False
    visual_outs.requires_grad = False
    text_ins.requires_grad = False
    text_outs.requires_grad = False
    
    # from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask
    temp_s=torch.ones(1,text_ins.shape[1])
    causal_attention_mask = _create_4d_causal_attention_mask(
        temp_s.size(), text_ins.dtype, device=device
    )

    #visual encoder part:
    for index_i, layer in enumerate(model.vision_model.encoder.layers):
        # layer.to(args.device)
        sparsegpt_prune_layer(layer,args,visual_ins, visual_outs, None, None,index_i,'visual')
        visual_ins,visual_outs=visual_outs,visual_ins
        torch.cuda.empty_cache()
    
    
    #text encoder part
    for index_i, layer in enumerate(model.text_model.encoder.layers):
        # layer.to(args.device)
        sparsegpt_prune_layer(layer,args,text_ins, text_outs, attention_mask, causal_attention_mask,index_i,'text')
        text_ins,text_outs=text_outs,text_ins
        torch.cuda.empty_cache()
        # layer.to('cpu')
    
    layer_inputs.clear()
    layer_outputs.clear() 
    return model

