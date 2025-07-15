import os
import numpy as np
import torch
from tqdm import tqdm

dataset = "ICE14-IMG-TXT"
entity_path = "../data/" + dataset + "/entity2id.txt"
entity_fig_matrix_list = []
with open(entity_path, "r", encoding="utf-8") as fr:
    for line in fr:
        entity_str = str(line.split()[0])
        entity = entity_str.strip('<').strip('>').strip('.').replace('\\', '/').strip('?').replace('\"', '')
        entity_fig_matrix_list.append("../data/" + dataset + "/fig/" + entity + "/matrix_visual.npy")
fig_embs_path = "../data/" + dataset + "/pre_train"
if not os.path.exists(fig_embs_path):
    os.makedirs(fig_embs_path)

if __name__ == "__main__":
    vector_list = []
    for vector_path in tqdm(entity_fig_matrix_list, desc="Processing visual vectors", unit="vector"):
        vector = torch.from_numpy(np.load(vector_path))
        vector_list.append(vector)
    matrix = torch.cat(vector_list, dim=1)
    print(matrix.shape)
    np.save("../pretrain/" + dataset + "/matrix_visual_vgg19.npy", matrix)
