## ReCo: Reliable Causal Chain Reasoning via Structural Causal Recurrent Neural Networks
For more details, please refer to our paper: [ReCo: Reliable Causal Chain Reasoning via Structural Causal Recurrent Neural Networks](https://aclanthology.org/2022.emnlp-main.431/).

### 1. Project Structure
```shell
--data
  |__ chinese # Chinese causal chain reasoning dataset
  |__ english # English causal chain reasoning dataset
--model
  |__ main_model.py # The definition of ReCo
--utils
  |__ tools.py # Data processing, tokenization and evaluation
--train.py # Python script for training ReCo
--train.sh # Shell script for training ReCo
```

### 2. Training
```shell
sh train.sh
```

### 3. Citation
If you want to cite our dataset and paper, you can use this BibTex:
```
@inproceedings{xiong-etal-2022-reco,
    title = "{R}e{C}o: Reliable Causal Chain Reasoning via Structural Causal Recurrent Neural Networks",
    author = "Xiong, Kai  and
      Ding, Xiao  and
      Li, Zhongyang  and
      Du, Li  and
      Liu, Ting  and
      Qin, Bing  and
      Zheng, Yi  and
      Huai, Baoxing",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    pages = "6426--6438"
}
```



