<div align="center">
<h1>Slide-CLIP: A Simple and Effective Pruning Method for CLIP</h1>
</div>

<div align="center">
IJCNN 2025 | [<a href="https://github.com/openai/CLIP">Paper</a>] | [<a href="https://github.com/openai/CLIP">Code</a>]
</div>


**What is it**: Slide-CLIP is the first task-specific CLIP pruning method that is effective, resources-saving, and easily expandable comparing to the previous pruning methods of CLIP.
Extensive experiments on CLIP-B-32, CLIP-L-14 and on datasets including MNIST, SVHN, CARS, CIFAR10, CIFAR100, GTSRB, FLOWER102, FOOD101, EUROSAT, SUN397, RESISC45 show the effectiveness of Slide-CLIP, outperforming competing baseline methods.

### 🏃 Requirements
The code is tested on following environment:
```
cuda==12.4
pytorch==2.4.1
python==3.10.12
transformers==4.47.1
datasets==3.0.1
numpy==1.26.0
```

### 🚀 Experiments

* Dataset 

    The dataset can be found on Huggingface.

* Model  

    The model can be found on Huggingface.

* Compression

    1.Firstly change the finetuned model path in all_prune_main.py.

    2.Secondly change the dataset path in data.py correspondingly.

    3.Thirdly change the following parameters in all_prune_main.py:
        
        sparsity_type: sparsity_type
        param_range: sparsity level
        target_dataset_list: the dataset used for model finetuning
        batch_size_list: calibration size for LSD
        choice_list: pruning method choice

    4.Lastly run the following command.
    ```bash
    python  all_prune_main.py
    ```


### 🌲 Expected Folder Structures

```
├── all_prune_main.py
├── lib
│   ├── data.py
│   ├── layerwrapper.py
│   ├── magnitude.py
│   ├── sliding_gpt.py
│   ├── sparsegpt.py
│   └── wanda.py
├── my_zero_shot.py
└── README.md                       
```

### 💬 Acknowledgments
This code is built upon <a href="https://github.com/openai/CLIP">CLIP</a>, <a href=https://github.com/IST-DASLab/sparsegpt>SparseGPT</a> and <a href=https://github.com/locuslab/wanda>Wanda</a>. Thanks for these awesome open-source projects!


### ✨ Citation
If you find our work or this code useful, please consider citing the corresponding paper:
```bibtex
@InProceedings{IJCNN-slide-clip,
  title = {Slide-CLIP: A Simple and Effective Pruning Method for CLIP},
  author = {Shi, Jiaxin and Luo, xin and S.Kevin Zhou},
  booktitle={2025 International Joint Conference on Neural Networks (IJCNN)}, 
  year = {2025}
}
```

