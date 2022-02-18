# Comparison-of-Encoders-on-SLU-tasks

Independent study in MIULAB during 2021/09-2022/01

- [Final Presentation Link](https://docs.google.com/presentation/d/18nfk_ufsX_xJxf5y5IPUaSt4JCpICYTT-2o4CNoqWis/edit#slide=id.g10965d62795_0_41)
- [Google colab Link](https://colab.research.google.com/drive/1Fa5BIlsSgCLfBUT6AOytyeCOLYCe5xCp?usp=sharing)

## User Instructions

Users only have to modify the parameters in the **Configuration** section in `Comparison-of-Encoders-on-SLU-tasks.ipynb`.

1. Download the dataset.

- [Fluent Speech](https://fluent.ai/fluent-speech-commands-a-dataset-for-spoken-language-understanding-research/)
- [ASRGLUE](https://drive.google.com/drive/folders/1slqI6pUiab470vCxQBZemQZN-a_ssv1Q)
- Move `asrglue_label.zip` to `ASRGLUE/dev` and unzip.
    - In this project we only use the audio in `ASRGLUE/dev`.<br>Since there is no audio in `ASRGLUE/train` and we don't have time to tidy up the label of `ASRGLUE/test`.


2. Modify the path.

- `BaseCFG.project_root`
    - The definite path of the directory to store checkpoints.
- `FluentSpeechCFG.data_root`
    - The definite path of the directory of `FluentSpeech` dataset.
- `AsrglueCFG.data_root`
    - The definite path of the directory of `ASRGLUE/dev` dataset.
    
3. Modify the parameters

- `curCFG`, `do_train`, `do_test`

## Intention

Same as [SLUE](https://arxiv.org/abs/2111.10367) says, "high-level" SLU tasks receives less attention since the speech-to-text data for SLU tasks is relatively difficult to collect. 

Hence, the ability of model to finetune on low-resource data becomes more important. If a pre-trained model captures the structure of speech well, then it should require few labeled examples to fine-tune it for SLU tasks.

It invokes our interest in comparing the performance of different encoders on low-resource, high-level SLU tasks.

## Process 

- Model
    - HuBERT (pretrained on LibriSpeech of 960hrs)
    - wav2vec2.0 (pretrained on LibriSpeech of 960hrs)
    - BERT (BERTForSequenceClassification)
- Dataset
    - ASRGLUE
    - FluentSpeech


Our final goal is to compare HuBERT and wav2vec2.0 on ASRGLUE dataset, and use BERT running on the text-transcript of the dev-set audio as the upper-baseline.

Before we finetune the models on ASRGLUE, we first finetune them on FluentSpeech and obtain the following result.

- HuBERT
valid_acc=0.98 
test_acc: 0.9965726337991037
- wav2Vec2.0
valid_acc=0.977
test acc = 0.9965726337991037

They both reaches 99% accuracy on FluentSpeech. It shows that our model is good enough, so the result on ASRGLUE is reliable.


## Experiment

### Dataset: ASRSLUE

#### Data Amount
- Use the audio in dev-set with 6 speakers.
![](https://i.imgur.com/FWYwJkY.png)

#### Split
- Train : Valid : Test = 0.72 : 0.18: 0.1
- For binary classification tasks (sst-2, qqp, scitail)
  - Make the # of 2 labels in test set same
      - If the model doesn't work, it gets accuracy 50%. (Just guess the same label for all input.)

### Findings-1

- We first consider every audio as a single data. So speaker1’s speech and speaker6’s speech to the same transcription may be in different set.
    - The test accuracy on every binary classification tasks exceeds 0.94
    - Even sts-b task has accuracy 0.85989004
    - It shows that **both of them has good capability of audio speech recognization**.
- So we resplited the test set.
    - **The audio with same transcription will be in the same set**.
- For binary classification tasks, test dataset has the same number of both labels to detect training failure.

### Result

![](https://i.imgur.com/3Cl593l.png)

### Findings-2

- Scitail performs the best considering all noise levels
    - It might because scientific proprietary nouns share similar prefix / postfix. So the model learns it better.
- HuBERT wins wav2vec2.0 on Sentiment Analysis tasks
    - This is mentioned in the SLUE paper.
    - Our result shows the same thing on sst-2 task.
- **HuBERT can be fine-tuned on understanding task easier**.
    - We tried hard to train wav2vec2.0
- HuBERT is able to fine-tuned even with batch-size=1
- Noise level does not have direct relation with better performance.

### Difficulties & Future Plan

- Due to CUDA memory constraint, we were unable to try larger batch size.
- Mixed the noise level of training dataset and testing dataset.
- Try to increase the number of data by processing the test set of ASR GLUE.



