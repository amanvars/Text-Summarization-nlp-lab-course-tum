import os
import argparse
from text_summarizer import TextSummarizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for text summarizer')
    parser.add_argument('--model', dest='model_type', type=str, help='Model type', choices=('bart', 'pegasus'),required=True)
    parser.add_argument('-d', '--dataset', dest='dataset_name', type=str, help='Dataset name', choices=('samsum', 'dialogsum', 'crd3'), required=True)
    parser.add_argument('--lr', help='Learning rate', type=float, default=3e-5)
    parser.add_argument('--epochs', dest='num_train_epochs', help='Number of training epochs', type=int, default=1)
    parser.add_argument('--bs', dest='batch_size', help='Batch size for training', type=int, default=8)
    parser.add_argument('--eval_bs', dest='eval_batch_size', help='Batch size for evaluation', type=int, default=2)
    parser.add_argument('--weight_decay', help='Weight decay', type=float, default=0.01)
    parser.add_argument('--save_total_limit', help='Save total limit', type=int, default=1)
    parser.add_argument('--patience', dest='early_stopping_patience', help='Patience for early stopping', type=float, default=0.1)
    parser.add_argument('--threshold', dest='early_stopping_threshold', help='Threshold for early stopping', type=int, default=10000)
    parser.add_argument('--max_input_length', help='Maximum length of the input', type=int, default=128)
    parser.add_argument('--max_target_length', help='Maximum length of the target', type=int, default=64)
    parser.add_argument('--save_path', help='Save path', type=str, default=os.getcwd()+'./models')

    args = parser.parse_args()
    kwargs = dict(args._get_kwargs())

    model = TextSummarizer(**kwargs)
    model.evaluate(kwargs['dataset_name'])