# Abstractive Text Summarization: A Comparison of ROUGE Scores and Human Evaluation
**Abstract.** The internet has resulted in an enormous amount of data generation, particularly in the textual form. Web pages, news articles, status updates, blogs, and other sources generate a lot of text data. However, this data is highly unstructured and is only useful for searching in documents or web pages. There is a great need to transform this text data into shorter, focused summaries which capture the necessary details. This leads to easier navigation of documents containing huge amounts of text. Yet, manually summarizing text is not possible on large-scale data. The distillation of a source document into a summary with the salient details can be automated. As such, the goal of automatically creating summaries of text is to have good generalization abilities of the method compared to humans. We aim to address this issue by evaluating the generated summary using ROUGE metric. Furthermore, we analyze the ROUGE score by performing manual evaluation and present our findings.

## Datasets

- SAMSUM from Gliwa et al.: [SAMSum Corpus: A Human-annotated Dialogue Dataset for Abstractive Summarization](https://aclanthology.org/D19-5409/)
- DialogSum from Chen et al.: [DialogSum: A Real-Life Scenario Dialogue Summarization Dataset](https://openreview.net/forum?id=v7CNycdHg3p)
- CRD3 from Rameshkumar and Bailey: [Storytelling with Dialogue: A Critical Role Dungeons and Dragons Dataset](https://aclanthology.org/2020.acl-main.459/?ref=https://githubhelp.com)

## Models
- BART from Lewis et al.:[BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://ai.facebook.com/research/publications/bart-denoising-sequence-to-sequence-pre-training-for-natural-language-generation-translation-and-comprehension/)
- Pegasus from Zhang et al.:[PEGASUS: A State-of-the-Art Model for Abstractive Text Summarization](http://proceedings.mlr.press/v119/zhang20ae).

## Installation
Create a virtual environment (conda is recommended). We used Python 3.7 in this project. Newer versions can also be installed.
```bash
conda create -n textsum python=3.7
```
Activate the environment:
```bash
conda activate textsum
```
Install the dependencies:
```bash
pip install -r requirements.txt
```

## Quick setup
Run the `main.py` file specifying the model and the dataset to evaluate on. For example:
```bash
python ./main.py --model bart --dataset dialogsum
```
Additional arguments can be provided (e.g. learning rate, batch size etc.). See `main.py` for details.

