## ReCo: Reliable Causal Chain Reasoning via Structural Causal Recurrent Neural Networks
For more details, please refer to our paper: [ReCo: Reliable Causal Chain Reasoning via Structural Causal Recurrent Neural Networks](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.431/).

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
```json
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
    url = "https://aclanthology.org/2022.emnlp-main.431",
    pages = "6426--6438",
    abstract = "Causal chain reasoning (CCR) is an essential ability for many decision-making AI systems, which requires the model to build reliable causal chains by connecting causal pairs. However, CCR suffers from two main transitive problems: threshold effect and scene drift. In other words, the causal pairs to be spliced may have a conflicting threshold boundary or scenario.To address these issues, we propose a novel Reliable Causal chain reasoning framework (ReCo), which introduces exogenous variables to represent the threshold and scene factors of each causal pair within the causal chain, and estimates the threshold and scene contradictions across exogenous variables via structural causal recurrent neural networks (SRNN). Experiments show that ReCo outperforms a series of strong baselines on both Chinese and English CCR datasets. Moreover, by injecting reliable causal chain knowledge distilled by ReCo, BERT can achieve better performances on four downstream causal-related tasks than BERT models enhanced by other kinds of knowledge.",
}
```



