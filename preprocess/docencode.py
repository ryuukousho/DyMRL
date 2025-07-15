import torch
import argparse
from pytorch_transformers import BertModel, BertConfig, BertTokenizer
from pytorch_transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from transformers import T5Tokenizer
import numpy as np

import warnings
warnings.filterwarnings(action='ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='linguistic')
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--plm", type=str, required=True)

    args = parser.parse_args()
    entity_path = "../data/" + args.dataset + "/entity2id.txt"
    entity_path_list = []
    with open(entity_path, "r", encoding="utf-8") as fr:
        for line in fr:
            entity_str = str(line.split()[0])
            entity = entity_str.strip('<').strip('>').strip('.').replace('\\', '/').strip('?').replace('\"', '')
            entity_path_list.append("../data/" + args.dataset + "/txt/" + entity)

    texts = []
    for entity_txt_path in entity_path_list:
        with open(entity_txt_path, 'r', encoding='utf-8') as entity_read:
            entity_txt_desc = entity_read.readline()
            texts.append(entity_txt_desc)
    # with open(entity_path, 'r', encoding='utf-8') as entity_read:
    #     for line in entity_read:
    #         line_entity = line.split('\t')[0].replace("_", " ").strip("<").strip(">")
    #         texts.append(line_entity)

    vocab_file = '../plms/' + args.plm + '/vocab.json'
    merges_file = '../plms/' + args.plm + '/merges.txt'
    tokenizer = RobertaTokenizer(vocab_file, merges_file)
    # tokenizer = BertTokenizer.from_pretrained('../plms/' + args.plm + '/vocab.txt')
    tokens, segments, input_masks = [], [], []
    for text in texts:
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens.append(indexed_tokens)
        segments.append([0] * len(indexed_tokens))
        input_masks.append([1] * len(indexed_tokens))
    max_len = max([len(single) for single in tokens])
    for j in range(len(tokens)):
        padding = [0] * (max_len - len(tokens[j]))
        tokens[j] += padding
        segments[j] += padding
        input_masks[j] += padding
    seg_len = len(tokens) // 20
    modelConfig = BertConfig.from_pretrained('../plms/' + args.plm + '/config.json')
    textExtractor = BertModel.from_pretrained('../plms/' + args.plm + '/pytorch_model.bin', config=modelConfig).to(
        torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    # textExtractor = RobertaModel.from_pretrained('../plms/' + args.plm + '/').to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    text_embeddings = None

    with torch.no_grad():
        for i_split in range(20):
            if i_split == 19:
                tokens_temp = tokens[i_split * seg_len: len(tokens)]
                segments_temp = segments[i_split * seg_len: len(tokens)]
                input_masks_temp = input_masks[i_split * seg_len: len(tokens)]
            else:
                tokens_temp = tokens[i_split * seg_len: (i_split + 1) * seg_len]
                segments_temp = segments[i_split * seg_len: (i_split + 1) * seg_len]
                input_masks_temp = input_masks[i_split * seg_len: (i_split + 1) * seg_len]
            if i_split == 0:
                tokens_tensor = torch.tensor(tokens_temp).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                segments_tensors = torch.tensor(segments_temp).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                input_masks_tensors = torch.tensor(input_masks_temp).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                output = textExtractor(tokens_tensor, segments_tensors, input_masks_tensors)
                text_embeddings = output[0][:, 0, :]
            else:
                tokens_tensor = torch.tensor(tokens_temp).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                segments_tensors = torch.tensor(segments_temp).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                input_masks_tensors = torch.tensor(input_masks_temp).to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                output = textExtractor(tokens_tensor, segments_tensors, input_masks_tensors)
                text_embeddings_temp = output[0][:, 0, :]
                text_embeddings = torch.cat((text_embeddings, text_embeddings_temp), dim=0)
    text_embeddings = text_embeddings.to(torch.device("cpu"))
    print(text_embeddings.shape)
    np.save("../data/" + args.dataset + "/pre_train/matrix_linguistic_" + args.plm + ".npy", text_embeddings)