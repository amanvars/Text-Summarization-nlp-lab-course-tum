import os
import numpy as np
import torch
import datasets
import statistics
import nltk
from rouge_score  import rouge_scorer
from transformers import pipeline
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer


class TextSummarizer:
    """
    Text summarization model for samsum, dialogsum and crd3 datasets
    """    
    def __init__(self, **kwargs):
        """Initialize the model""" 

        self.lr=kwargs['lr']
        self.batch_size=kwargs['batch_size']
        self.eval_batch_size=kwargs['eval_batch_size']
        self.weight_decay=kwargs['weight_decay']
        self.save_total_limit=kwargs['save_total_limit']
        self.num_train_epochs=kwargs['num_train_epochs']
        self.early_stopping_patience=kwargs['early_stopping_patience']
        self.early_stopping_threshold=kwargs['early_stopping_threshold']
        self.metric=datasets.load_metric('rouge')
        self.prefix = "summarize: "
        self.max_input_length = kwargs['max_input_length']
        self.max_target_length = kwargs['max_target_length']
        self.scorer = rouge_scorer.RougeScorer(['rouge1','rouge2' ,'rougeL','rougeLsum'], use_stemmer=True)
        self.prediction=[]
        self.rouge1_scores_lst=[]
        self.rouge2_scores_lst=[]
        self.rougeL_scores_lst=[]
        self.rougeLsum_scores_lst=[]
        
        if kwargs['model_type'].lower() == 'bart':
            self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
        else:
            self.tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-cnn_dailymail")
            self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-cnn_dailymail")

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model) # create a batch of examples
        self.save_path = kwargs['save_path']
        os.makedirs(os.getcwd()+'/'+self.save_path, exist_ok=True)
  
    def evaluate(self, dataset_name: str):
        """Select the dataset to be used for training and evaluation

        Args:
            dataset_name (str): name of the dataset

        Returns:
            str or None: evaluation result
        """        

        self.dataset_name=dataset_name.lower()
        if self.dataset_name=="samsum":
            self.samsum()
        elif self.dataset_name=="dialogsum":
            self.dialogsum()
        elif self.dataset_name=="crd3":
            self.crd3()
        else:
            print("Please select from available datasets name: samsum , dialogsum  &  crd3")
            return None

    def preprocess_function(self, examples):
        """Preprocess the examples for the model

        Args:
            examples (dict): dictionary of examples

        Returns:
            dict: dictionary of preprocessed examples
        """   

        target_label="summary"
        if self.dataset_name=="crd3":
            examples["dialogue"]=['\n'.join([f"""{x["names"][0]} : {" ".join(x["utterances"])}""" for x in y]) for y in examples["turns"]]
            target_label="chunk"

        inputs=[self.prefix + x for x in examples["dialogue"]]
        model_inputs = self.tokenizer(inputs, max_length=self.max_input_length, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(examples[target_label], max_length=self.max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def compute_metrics(self,eval_pred):
        """Compute the metrics for the model

        Args:
            eval_pred (tuple): tuple of predictions and labels 

        Returns:
            dict: dictionary of metrics
        """  

        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them, replace with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence, so split text into sentence and add new line character
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True) #use stem form of words
        # Extract rouge fmeasure*100 results for rouge1,rouge2,rougel,rougelsum
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions] # checking for each prediction number of non pad tokens
        result["gen_len"] = np.mean(prediction_lens) # take average generated length
        
        return {k: round(v, 4) for k, v in result.items()} #all rouge fmeasure*100 and average length

    def training(self):
        """Train the model"""   

        tokenized_datasets = self.tokenized_datasets.with_format("torch")
        training_args = Seq2SeqTrainingArguments(
            output_dir=os.getcwd() + "/" + self.save_path,
            evaluation_strategy="epoch",
            learning_rate=self.lr,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            num_train_epochs=self.num_train_epochs,
            fp16=True,
            eval_accumulation_steps=64,
            save_strategy="epoch",
            metric_for_best_model="eval_rouge1",
            predict_with_generate=True,
            greater_is_better=True,
            seed=42,
            generation_max_length=self.max_target_length,load_best_model_at_end=True
        )

        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,compute_metrics=self.compute_metrics,
            callbacks = [EarlyStoppingCallback(early_stopping_patience = self.early_stopping_patience,
                        early_stopping_threshold=self.early_stopping_threshold)]
        )

        self.trainer.train()

    def samsum(self):
        """Train and evaluate the model on samsum dataset"""

        test_dataset=datasets.load_dataset("samsum",split="test")
        summarizer = pipeline("summarization", model="philschmid/bart-large-cnn-samsum")
        predict_summary=summarizer(test_dataset["dialogue"])
        i=0
        for prediction,summary in zip(predict_summary,test_dataset["summary"]):
            i+=1
            if i%100==0:
                print(f"{i}th sample done")

        prediction=prediction["summary_text"]
        prediction="\n".join(nltk.sent_tokenize(prediction.strip()))
        self.prediction.append(prediction)
        scores = self.scorer.score(prediction,summary)
        self.rouge1_scores_lst.append(scores["rouge1"].fmeasure)
        self.rouge2_scores_lst.append(scores["rouge2"].fmeasure)
        self.rougeL_scores_lst.append(scores["rougeL"].fmeasure)
        self.rougeLsum_scores_lst.append(scores["rougeLsum"].fmeasure)
        
        print("rouge1",statistics.mean(self.rouge1_scores_lst))
        print("rouge2",statistics.mean(self.rouge2_scores_lst))
        print("rougeL",statistics.mean(self.rougeL_scores_lst))
        print("rougeLsum",statistics.mean(self.rougeLsum_scores_lst))

    def dialogsum(self):
        """Train and evaluate the model on dialogsum dataset"""

        rouge1_scores_lst_summary1=[]
        rouge2_scores_lst_summary1=[]
        rougeL_scores_lst_summary1=[]
        rougeLsum_scores_lst_summary1=[]
        rouge1_scores_lst_summary2=[]
        rouge2_scores_lst_summary2=[]
        rougeL_scores_lst_summary2=[]
        rougeLsum_scores_lst_summary2=[]
        rouge1_scores_lst_summary3=[]
        rouge2_scores_lst_summary3=[]
        rougeL_scores_lst_summary3=[]
        rougeLsum_scores_lst_summary3=[]
        self.datasets=datasets.load_dataset("knkarthick/dialogsum")
        self.tokenized_datasets=self.datasets.map(self.preprocess_function,batched=True)
        self.training()
        dialogSum_test=datasets.load_dataset("json",data_files="/content/gdrive/MyDrive/Dialogsum_test.json")
        i=0
        for inputs,summary1,summary2,summary3 in zip(dialogSum_test["train"]["dialogue"],dialogSum_test["train"]["summary1"],dialogSum_test["train"]["summary2"],dialogSum_test["train"]["summary3"]):
            
            try:
                i+=1
                inp = self.tokenizer(inputs, return_tensors="pt").to(self.device)
                outputs = self.trainer.model.generate(**inp)
                pred=np.array(outputs.cpu()[0])
                gen_sum=self.tokenizer.decode(pred,skip_special_tokens=True, clean_up_tokenization_spaces=False)
                self.prediction.append(gen_sum)
                scores1=self.scorer.score(gen_sum,summary1)
                scores2=self.scorer.score(gen_sum,summary2)
                scores3=self.scorer.score(gen_sum,summary3)

                rouge1_scores_lst_summary1.append(scores1["rouge1"].fmeasure)
                rouge2_scores_lst_summary1.append(scores1["rouge2"].fmeasure)
                rougeL_scores_lst_summary1.append(scores1["rougeL"].fmeasure)
                rougeLsum_scores_lst_summary1.append(scores1["rougeLsum"].fmeasure)

                rouge1_scores_lst_summary2.append(scores2["rouge1"].fmeasure)
                rouge2_scores_lst_summary2.append(scores2["rouge2"].fmeasure)
                rougeL_scores_lst_summary2.append(scores2["rougeL"].fmeasure)
                rougeLsum_scores_lst_summary2.append(scores2["rougeLsum"].fmeasure)

                rouge1_scores_lst_summary3.append(scores3["rouge1"].fmeasure)
                rouge2_scores_lst_summary3.append(scores3["rouge2"].fmeasure)
                rougeL_scores_lst_summary3.append(scores3["rougeL"].fmeasure)
                rougeLsum_scores_lst_summary3.append(scores3["rougeLsum"].fmeasure)
                if i % 100 == 0:
                    print(f"{i}th sample done")
            except:
                pass

            self.rouge1_scores_lst=((np.array(rouge1_scores_lst_summary1)+np.array(rouge1_scores_lst_summary2)+np.array(rouge1_scores_lst_summary3))/3).tolist()
            self.rouge2_scores_lst=((np.array(rouge2_scores_lst_summary1)+np.array(rouge2_scores_lst_summary2)+np.array(rouge2_scores_lst_summary3))/3).tolist()
            self.rougeL_scores_lst=((np.array(rougeL_scores_lst_summary1)+np.array(rougeL_scores_lst_summary2)+np.array(rougeL_scores_lst_summary3))/3).tolist()
            self.rougeLsum_scores_lst=((np.array(rougeLsum_scores_lst_summary1)+np.array(rougeLsum_scores_lst_summary2)+np.array(rougeLsum_scores_lst_summary3))/3).tolist()

            print("rouge1",statistics.mean(self.rouge1_scores_lst))
            print("rouge2",statistics.mean(self.rouge2_scores_lst))
            print("rougeL",statistics.mean(self.rougeL_scores_lst))
            print("rougeLsum",statistics.mean(self.rougeLsum_scores_lst))

    def crd3(self):
        """Train and evaluate the model on crd3 dataset"""        
        self.datasets=datasets.load_dataset("crd3",revision="main")
        self.tokenized_datasets=self.datasets.map(self.preprocess_function,batched=True)
        self.training()
        self.prediction=[]
        i=0
        for inputs,summary in zip(self.tokenized_datasets["test"]["dialogue"],self.tokenized_datasets["test"]["chunk"]):
            try:
                i+=1
                inp = self.tokenizer(inputs, return_tensors="pt").to(self.device)
                outputs = self.trainer.model.generate(**inp)
                pred=np.array(outputs.cpu()[0])
                self.prediction.append(pred)
                gen_sum=self.tokenizer.decode(pred,skip_special_tokens=True, clean_up_tokenization_spaces=False)
                scores=self.scorer.score(gen_sum,summary)
        

                self.rouge1_scores_lst.append(scores["rouge1"].fmeasure)
                self.rouge2_scores_lst.append(scores["rouge2"].fmeasure)
                self.rougeL_scores_lst.append(scores["rougeL"].fmeasure)
                self.rougeLsum_scores_lst.append(scores["rougeLsum"].fmeasure)
                if i%100==0:
                    print(f"{i}th sample done")
            except:
                continue

            print("rouge1",statistics.mean(self.rouge1_scores_lst))
            print("rouge2",statistics.mean(self.rouge2_scores_lst))
            print("rougeL",statistics.mean(self.rougeL_scores_lst))
            print("rougeLsum",statistics.mean(self.rougeLsum_scores_lst))