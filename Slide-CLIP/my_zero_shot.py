import ssl
ssl._create_default_https_context=ssl._create_unverified_context
import torch
import copy
from torch import nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from lib.data import cifar100_templates,flower102_templates,eurosat_templates,food101_templates,sun397_templates,oxford_pets_templates,resisc45_templates,country211_templates,cifar10_templates,gtsrb_templates,cars_templates,mnist_templates,svhn_templates
from lib.data import cifar100_class_map,flower102_classes_map,food101_classes_map,eurosat_classes_map,sun397_classes_map,oxford_pets_classes_map,resisc45_classes_map,country211_classes_map,cifar10_classes_map,gtsrb_classes_map,cars_classes_map,mnist_classes_map,svhn_classes_map
from lib.data import get_split_loader


def zero_shot_classifier(model, classnames, templates,tokenizer,device):
    padding_classnames = copy.deepcopy(classnames)
    def _get_classname_emb(classname):
        texts = [template.format(classname) if isinstance(template, str) else template(
            classname) for template in templates]  # format with class
        textinputs = tokenizer(texts, padding=True, return_tensors="pt")
        textinputs = textinputs.to(device)
        class_embeddings = model.get_text_features(**textinputs)
        class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
        class_embedding /= class_embedding.norm()
        return class_embedding

    with torch.no_grad():
        zeroshot_weights = []
        for classname in padding_classnames:
            class_embedding = _get_classname_emb(classname)
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)

    return zeroshot_weights

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]

def run(model, classifier, dataloader,device):
    total_batch_size = dataloader.batch_size
    # print(f"total_batch_size:{total_batch_size}")
    n=0.
    with torch.no_grad():
        top1, top5 = 0., 0.
        bar = tqdm(dataloader, unit_scale=total_batch_size)
        for batch in bar:
            images=batch['pixel_values']
            batch_size = images.shape[0]
            target=batch["labels"]
    
            images=images.reshape(batch_size,3,224,224)
            images = images.to(device)
            target = target.to(device)
            image_features = model.get_image_features(images)
            image_features = F.normalize(image_features, dim=-1)
            logits = 100. * image_features @ classifier
            
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            bar.set_description(
                f'Acc@1 {acc1 / batch_size:.3f} Acc@5 {acc5 / batch_size:.3f}')
            top1 += acc1
            top5 += acc5
            n += batch_size
            del images, target, logits

    return float(top1)/float(n), float(top5)/float(n)

def zero_shot_eval(model,args,tokenizer=AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32"),device='cuda:5'):
    model.to(device)
    model.eval()
    top1, top5=0,0
    print(f'Starting zero-shot on dataset:{args.data_name}')

    match args.data_name:
        case "cifar100":
            class_names=list(cifar100_class_map.values())
            prompt_template=cifar100_templates
        case "flower102":
            class_names=list(flower102_classes_map.values())
            prompt_template=flower102_templates
        case "food101":
            class_names=list(food101_classes_map.values())
            prompt_template=food101_templates
        case "eurosat":
            class_names=list(eurosat_classes_map.values())
            prompt_template=eurosat_templates
        case "sun397":
            class_names=list(sun397_classes_map.values())
            prompt_template=sun397_templates
        case "oxford_pets":
            class_names=list(oxford_pets_classes_map.values())
            prompt_template=oxford_pets_templates
        case "resisc45":
            class_names=list(resisc45_classes_map.values())
            prompt_template=resisc45_templates  
        case "country211":
            class_names=list(country211_classes_map.values())
            prompt_template=country211_templates      
        case "cifar10":
            class_names=list(cifar10_classes_map.values())
            prompt_template=cifar10_templates      
        case "gtsrb":
            class_names=list(gtsrb_classes_map.values())
            prompt_template=gtsrb_templates   
        case "cars":
            class_names=list(cars_classes_map.values())
            prompt_template=cars_templates  
        case "mnist":
            class_names=list(mnist_classes_map.values())
            prompt_template=mnist_templates 
        case "svhn":
            class_names=list(svhn_classes_map.values())
            prompt_template=svhn_templates 
        case _:
            raise RuntimeError("Not support dataset name !")    
             
    classifier = zero_shot_classifier(
        model, class_names, prompt_template,tokenizer,device)

    data_loader=get_split_loader('test', args.data_name, args.test_batch_size, args)
    top1, top5 = run(model, classifier,data_loader,device)
    print(f"Acc@1/Acc@5:{top1:.3f}/{top5:.3f}")
    print(f"finish zero shot\n")
    return top1,top5

