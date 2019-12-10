import numpy as np
import re
import torch
import preprocessor as p
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam, BertConfig
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm
from sklearn.metrics import f1_score
from enum import Enum
import csv
import random

CACHE = "./models"
BERT_MODEL = "bert-base-cased"
MAX_SEQ_LENGTH = 180

p.set_options(p.OPT.MENTION, p.OPT.EMOJI)

class Task(Enum):
    A = "A"


TASK_LABELS = {
    Task.A: {
        "NOT": 0,
        "OFF": 1
    }
}
TRAIN = "./data/olid-training-v1.0.tsv"
TEST = {
    Task.A: "./data/testset-levela.tsv"
}


def load_test_dataset(task):
    ids = []
    x = []
    with open(TEST[task], 'rt') as test:
        for inp in csv.reader(test, delimiter='\t'):
 #           print(inp[1])
            # Delete #s
            temp = inp[1].replace("#", "")
            # Delete URLs
            temp = temp.replace("URL", "")
            # Delete mentions & emojis
            temp = p.clean(temp)
            # Remove repeating characters
            temp = re.sub(r'(.)\1+', r'\1\1', temp)
#            print(temp)
            x.append(temp)
            ids.append(inp[0])
        return ids, x


def load_train_dataset(task):
    with open(TRAIN, 'rt') as entire_set:
        tsvin = csv.reader(entire_set, delimiter='\t')

        x = []
        y = []
        for i, inp in enumerate(tsvin):
            if i != 0:
                # Delete #s
                temp = inp[1].replace("#", "")
                # Delete URLs
                temp = temp.replace("URL", "")
                # Delete mentions & emojis
                temp = p.clean(temp)
                # Remove repeating characters
                temp = re.sub(r'(.)\1+', r'\1\1', temp)
                x.append(temp)
                y.append(TASK_LABELS[task][inp[2]])
        return np.array(x), np.array(y)


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def convert_examples_to_features(x, y, max_seq_length, tokenizer):
    features = []
    for i, x in enumerate(x):
        tokens = tokenizer.tokenize(x)
        if len(tokens) > max_seq_length - 2:
            tokens = tokens[:(max_seq_length - 2)]
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        label_id = y[i]

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_id=label_id))
    return features


class ClassificationModel:
    def __init__(self, task, val=0.1, bert_model=BERT_MODEL, gpu=True, seed=0):
        self.gpu = gpu
        self.task = task
        self.bert_model = bert_model
        x_total, y_total = load_train_dataset(self.task)
        nb_data = x_total.shape[0]
        self.x_train, self.y_train = x_total[:int(nb_data * (1-val))], y_total[:int(nb_data * (1-val))]
        self.x_test, self.y_test = x_total[int(nb_data * (1-val)):], y_total[:int(nb_data * (1-val))]
        # self.x_val = np.random.choice(self.x_train, size=(int(val * len(self.x_train)),), replace=False)
        # self.y_val = np.random.choice(self.y_train, size=(int(val * len(self.x_train)),), replace=False)
        self.x_test_ids, self.x_test = load_test_dataset(self.task)
        self.num_classes = len(TASK_LABELS[task])

        self.model = None
        self.optimizer = None
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)

        self.plt_x = []
        self.plt_y = []
        self.plt_y_val = []

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.gpu:
            torch.cuda.manual_seed_all(seed)

    def __init_model(self):
        if self.gpu:
            self.device = torch.device("cuda")
            print("Learning start with GPU")
        else:
            self.device = torch.device("cpu")
            print("Learning start with CPU")
        self.model.to(self.device)
        print(torch.cuda.memory_allocated(self.device))

    def new_model(self):
        self.model = BertForSequenceClassification.from_pretrained(self.bert_model, num_labels=self.num_classes)
        self.__init_model()

    def load_model(self, path_model, path_config):
        self.model = BertForSequenceClassification(BertConfig(path_config), num_labels=self.num_classes)
        self.model.load_state_dict(torch.load(path_model))
        self.__init_model()

    def save_model(self, path_model, path_config):
        torch.save(self.model.state_dict(), path_model)
        with open(path_config, 'w') as f:
            f.write(self.model.config.to_json_string())

    # noinspection PyArgumentList
    def train(self, epochs, plot_path, batch_size=32, lr=5e-5, model_path=None, config_path=None):
        model_params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in model_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1,
                                  t_total=int(len(self.x_train) / batch_size) * epochs)

        train_features = convert_examples_to_features(self.x_train, self.y_train, MAX_SEQ_LENGTH, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        _, counts = np.unique(self.y_train, return_counts=True)
        class_weights = [sum(counts) / c for c in counts]
        example_weights = [class_weights[e] for e in self.y_train]
        sampler = WeightedRandomSampler(example_weights, len(self.y_train))
        train_dataloader = DataLoader(train_data, sampler=sampler, batch_size=batch_size)

        self.model.train()
        for e in range(epochs):
            loss_sum = 0
            print("Epoch {e}".format(e=e))
            f1, acc = self.val()
            print("\nF1 score: {f1}, Accuracy: {acc}".format(f1=f1, acc=acc))
            # about val loss
            self.plt_x.append(e)
            self.plt_y_val.append(acc)
            if model_path is not None and config_path is not None:
                self.save_model(model_path, config_path)
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                loss.backward()
                print("step and epochs and batch_size", step, e, batch_size)
                loss_sum += loss.item()
                # self.plt_y.append(loss.item())
                # self.plt_x.append(nb_tr_steps)
                # self.save_plot(plot_path)

                # nb_tr_steps += 1
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.gpu:
                    torch.cuda.empty_cache()
            # self.plt_x.append(e) already append before
            self.plt_y.append(loss_sum)
            self.save_plot(plot_path)

    def val(self, batch_size=32, test=False):
        eval_features = convert_examples_to_features(self.x_val, self.y_val, MAX_SEQ_LENGTH, self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)

        f1, acc = 0, 0
        nb_eval_examples = 0

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            predicted_labels = np.argmax(logits.detach().cpu().numpy(), axis=1)
            acc += np.sum(predicted_labels == gnd_labels.numpy())
            tmp_eval_f1 = f1_score(predicted_labels, gnd_labels, average='macro')
            f1 += tmp_eval_f1 * input_ids.size(0)
            nb_eval_examples += input_ids.size(0)
        print("nb_eval_examples : %d"%nb_eval_examples)

        return f1 / nb_eval_examples, acc / nb_eval_examples

    def save_plot(self, path):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(self.plt_x, self.plt_y, 'r')
        ax.plot(self.plt_x, self.plt_y_val, 'g')

        ax.set(xlabel='Training steps', ylabel='Loss')

        fig.savefig(path)
        plt.close()

    def create_test_predictions(self, path):
        eval_features = convert_examples_to_features(self.x_test, [-1] * len(self.x_test), MAX_SEQ_LENGTH,
                                                     self.tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=16)

        predictions = []
        inverse_labels = {v: k for k, v in TASK_LABELS[self.task].items()}

        for input_ids, input_mask, segment_ids, gnd_labels in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask)

            predictions += [inverse_labels[p] for p in list(np.argmax(logits.detach().cpu().numpy(), axis=1))]
        with open(path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for i, prediction in enumerate(predictions):
                if i!=0:
                    writer.writerow([int(self.x_test_ids[i]), prediction])

        return predictions


if __name__ == "__main__":

    PLOT_PATH = "./plot.png"
    PATH_CONFIG = "./results/a-uncased-4-epochs/config"
    PATH_STATE = "./results/a-uncased-4-epochs/state"

    cm = ClassificationModel(Task.A, gpu='cuda', seed=0, val=0.20)
    cm.new_model()

    cm.train(epochs=8, plot_path=PLOT_PATH, batch_size=32, lr=5e-04, model_path=PATH_STATE, config_path=PATH_CONFIG)
    cm.create_test_predictions("./a_pred.csv")