import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import os
import numpy as np

vgg_model_4096 = models.vgg19(pretrained=True)
new_classifier = torch.nn.Sequential(*list(vgg_model_4096.children())[-1][:6])
vgg_model_4096.classifier = new_classifier
trans = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])


dataset = "GDELT-IMG-TXT"
entity_path = "../data/" + dataset + "/entity2id.txt"
entity_path_list = []
with open(entity_path, "r", encoding="utf-8") as fr:
    for line in fr:
        # entity_str = str(line.split()[0])
        rel, id = line.strip().split("\t")
        begin = rel.find('(')
        entity_str = rel[:begin].strip()
        entity = entity_str.strip('<').strip('>').strip('.').replace('\\', '/').strip('?').replace('\"', '')
        entity_path_list.append("../data/" + dataset + "/fig/" + entity)
fig_embs_path = "../data/" + dataset + "/pre_train"
if not os.path.exists(fig_embs_path):
    os.makedirs(fig_embs_path)

if __name__ == "__main__":
    final_fig_emb = []
    for entity in entity_path_list:
        print("Processing " + entity)
        fig_names = os.listdir(entity)
        fig_names = [file for file in fig_names if file.endswith('.jpg')]
        fig_emb_list = []
        for each_fig_name in fig_names:
            image_dir = entity + "/" + each_fig_name

            im = Image.open(image_dir)
            try:
                im = trans(im)
                im.unsqueeze_(dim=0)

                image_feature_4096 = vgg_model_4096(im).data[0]
                # print('dim of vgg_model_4096: ', image_feature_4096.shape)
                fig_emb_list.append(image_feature_4096)
            except Exception as e:
                # print(image_dir)
                continue
        if len(fig_emb_list) == 0:
            with open(fig_embs_path + "/error_stat.txt", "a", encoding="utf-8") as ft:
                ft.write(entity + "\n")
            continue
        fig_emb_sum = torch.cat(fig_emb_list).view(-1, len(fig_emb_list))
        fig_emb_mean = torch.mean(fig_emb_sum, dim=1, keepdim=True)
        print(fig_emb_mean.shape)
        # final_fig_emb.append(fig_emb_mean)
        np.save(entity + "/matrix_visual.npy", fig_emb_mean)
    # fig_embs = torch.cat(final_fig_emb, dim=1)
    # try:
    #     print("Storing fig embeddings with size:" + str(fig_embs.shape))
    # except Exception as e:
    #     pass
    # np.save(fig_embs_path + "matrices_visual.npy", fig_embs)
