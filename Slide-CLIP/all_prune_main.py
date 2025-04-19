import argparse
import ssl
import os 
ssl._create_default_https_context=ssl._create_unverified_context
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import torch
import sys
from importlib.metadata import version
from safetensors.torch import load_file
import copy
import time
from lib.slide_clip import prune_clip_with_sliding_sparsegpt,prune_clip_with_sparsegpt,second_prune_clip_with_sliding_sparsegpt
from lib.slide_clip import check_sparsity
from lib.wanda import prune_clip_with_wanda
from lib.magnitude import prune_clip_with_magnitude
from transformers import CLIPModel,AutoProcessor,AutoTokenizer
from my_zero_shot import zero_shot_eval

print('torch', version('torch'))
print('transformers', version('transformers'))
print('# of gpus: ', torch.cuda.device_count())
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='openai/clip-vit-base-patch32', choices=["openai/clip-vit-base-patch32", "openai/clip-vit-base-patch16", "openai/clip-vit-large-patch14"], type=str, help='LLaMA model')
    parser.add_argument('--data_name', default='cifar100',
                        help="{cifar100,flower102,food101,eurosat,sun397,oxford_pets,resisc45...}")
    parser.add_argument('--batch_size', default=128 ,type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--test_batch_size', default=200 ,type=int,
                        help='Size of a test mini-batch.')
    parser.add_argument('--split_batch_size', default=256 ,type=int,
                        help='Size of a test mini-batch.')
    parser.add_argument('--device', default="cuda:3", type=str,
                        help='device to run')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=256, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument('--imgsparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument('--textsparsity_ratio', type=float, default=0.8, help='Sparsity level')
    parser.add_argument("--sparsity_type", default="unstructured", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", default="sliding_sparsegpt", type=str)
    parser.add_argument('--tokenizer', default=AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32"),
                        help='tokenizer')
    parser.add_argument('--processor', default=AutoProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                        help='tokenizer')
    parser.add_argument('--prefix', default='unstructured',
                        type=str, choices=["unstructured", "4:8", "2:4",'mix_retrieve','retrieve','diff_batch_size_with_additional_loss',"debug","standard", "test","prompt_normal","prompt_random","prompt_random_with_adloss"])
    parser.add_argument('--midfix', default='progressive_standard',
                        type=str, choices=['with_additional_loss',"no_additional_loss"])
    parser.add_argument('--save_model', default=False, help='boolean value to decide if save model')
    parser.add_argument('--train_epoch_loss_threshold', type=float, default=0.01, help='train_epoch_loss_threshold')
    parser.add_argument('--train_epoch', type=int, default=5, help='train_epoch')
    parser.add_argument('--get_simi', type=bool, default=False, help='get_simi')
    parser.add_argument('--prune_n', type=int, default=0, help='prune_n')
    parser.add_argument('--prune_m', type=int, default=0, help='prune_m')
    parser.add_argument('--param_range', default=[0.5,0.6,0.7,0.8,0.9], help='sparsity_range')
    parser.add_argument('--step2_prune_ratio', type=int, default=0.7, help='target ratio')
    parser.add_argument('--prune_ratio', type=float, default=0.7, help='target ratio')
    parser.add_argument('--only_visual', type=int, default=0, help='only_visual')
    
    
    args = parser.parse_args()
    # args.prefix=args.sparsity_type
    if args.midfix=='progressive_standard':
        args.prefix='progressive_standard/'+args.prefix

   
    args.tokenizer=AutoTokenizer.from_pretrained(args.model,device=args.device)
    args.processor=AutoProcessor.from_pretrained(args.model,device=args.device)
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))
        args.prune_n=prune_n
        args.prune_m=prune_m

    path=f"/home/jiaxinshi/jiaxinshi/Slide-CLIP/{args.prefix}"
    if not os.path.exists(path):
        # 创建目录
        os.makedirs(path)
        print(f"目录 '{path}' 已创建。")
    else:
        print(f"目录 '{path}' 已存在。")
    
    
    #加载模型
    device = args.device
    print("use device ", device)
    
    processor = args.processor
    param_range=[0.8]
    if args.sparsity_type != "unstructured":
        param_range=[0.5]
    # target_dataset_list=['cifar100','flower102','food101','eurosat','sun397','resisc45','mnist','svhn','cars']
    target_dataset_list=['cifar100']
    args.prefix='8_dataset'

    # batch_size_list=[125,250,500,1000,2000,4000,8000]    
    batch_size_list=[250]    

    # choice_list=['sliding_sparsegpt_progress','sliding_sparsegpt',"sparsegpt", "wanda","magnitude"]
    choice_list=['sliding_sparsegpt_progress','sliding_sparsegpt']


    for sparsity_ratio in param_range:
        # sparsity_ratio=args.prune_ratio
        for choice in choice_list:
            args.prune_method=choice
            for batchsize in batch_size_list:
                dataset_list=copy.deepcopy(target_dataset_list)
                curbatch=batchsize
                model_name=args.model.split('/')
                filename=f'{path}/{model_name[1]}_{args.prefix}_{args.midfix}_{args.sparsity_type}_{args.prune_method}_{args.train_epoch}_{sparsity_ratio:.2%}_{curbatch}.log'

                start_time = time.time()
                with open(filename, 'w') as f:
                    # 重定向输出
                    sys.stdout = f
                    average_top1=0
                    average_atop5=0
                    istop1top5=False
                    for dataname in dataset_list:
                        args.batch_size=curbatch
                        args.nsamples=args.batch_size
                        args.split_batch_size=args.batch_size                           
                        args.data_name=dataname
                        # args.prune_method=choice
                        if dataname=='flower102' and batchsize>1000:
                            args.batch_size=1000
                            args.nsamples=args.batch_size
                            args.split_batch_size=args.batch_size
                        # for textsparsity_ratio in param_range:
                        model = CLIPModel.from_pretrained(args.model)

                        if '32' in model_name[1]:
                            finetuned_vision_encoder_path = f"/home/jiaxinshi/huggingface_model/clip-b32-finetune/{dataname}/model.safetensors"
                        if '14' in model_name[1]:
                            finetuned_vision_encoder_path = f"/home/jiaxinshi/huggingface_model/clip-l14-finetune/{dataname}/model.safetensors"

                        finetuned_state_dict = load_file(finetuned_vision_encoder_path)
                    
                        new_state_dict = {}
                        prefix = "vision_model."
                        for key in finetuned_state_dict.keys():
                            if key.startswith(prefix):
                                new_key = key[len(prefix):]
                                new_state_dict[new_key] = finetuned_state_dict[key]
                            else:
                                new_state_dict[key] = finetuned_state_dict[key]

                        # 更新模型的 Vision Encoder 部分
                        model.vision_model.load_state_dict(new_state_dict)
                        # model.to(device)
                        
                        model.eval()
                        args.imgsparsity_ratio=sparsity_ratio
                        args.textsparsity_ratio=sparsity_ratio
                        args.sparsity_ratio=sparsity_ratio
                        args.step2_prune_ratio=sparsity_ratio
                        print(f"pruning method:{args.prune_method}")
                        print(f"pruning starts \n   |sparsity_ratio:{args.sparsity_ratio} \n  |imgsparsity_ratio :{args.imgsparsity_ratio} \n   |textsparsity_ratio:{args.textsparsity_ratio}")
                        if args.prune_method=='wanda':
                            pruned_model=prune_clip_with_wanda(args, model, device=device, prune_n=prune_n, prune_m=prune_m,preprocess=processor)
                        elif args.prune_method=='sparsegpt':
                            pruned_model=prune_clip_with_sparsegpt(args, model, device=device, prune_n=prune_n, prune_m=prune_m,preprocess=processor)
                        elif args.prune_method=='magnitude':
                            pruned_model=prune_clip_with_magnitude(args, model)
                        elif args.prune_method=='sliding_sparsegpt':
                            pruned_model=prune_clip_with_sliding_sparsegpt(args, model, device=device, prune_n=prune_n, prune_m=prune_m,preprocess=processor)
                        elif args.prune_method=='sliding_sparsegpt_progress':
                            step1_ratio=0.5                               
                            args.imgsparsity_ratio=step1_ratio
                            args.textsparsity_ratio=step1_ratio
                            args.sparsity_ratio=step1_ratio
                            pruned_model=second_prune_clip_with_sliding_sparsegpt(args, model, device=device, prune_n=prune_n, prune_m=prune_m,preprocess=processor)
                        
                        if args.save_model:
                            save_name=f'./paper_pth/{model_name[1]}_{args.prefix}_{args.midfix}_{args.sparsity_type}_{choice}_{args.train_epoch_loss_threshold}_{sparsity_ratio:.2%}_{curbatch}.pth'
                            torch.save(pruned_model.state_dict(), save_name)
                        
                        pruned_model.to(device)
                        pruned_model.eval()
                        
                        check_sparsity(pruned_model)
                        torch.cuda.empty_cache()

                        top1,top5=zero_shot_eval(pruned_model,args=args,tokenizer=args.tokenizer,device=device)
                        if not isinstance(top5,dict):
                            average_top1+=top1
                            average_atop5+=top5
                            istop1top5=True
                            
                    if(istop1top5):
                        print(f"{choice}: average acc1/acc5:{average_top1/len(dataset_list):.3f}/{average_atop5/len(dataset_list):.3f}")
                    end_time = time.time()

                    # 计算并打印时间差（转换为毫秒）
                    elapsed_time = (end_time - start_time)   
                    print(f"excution time: {elapsed_time:.2f} s")        
                    print(f'pruning done')  
                # 恢复标准输出   
                sys.stdout = sys.__stdout__    



    
if __name__ == '__main__':
    main()