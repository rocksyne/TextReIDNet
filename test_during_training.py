# -*- coding: utf-8 -*-
import torch
import numpy as np
import os
from utils.read_write_data import write_txt
from tqdm import tqdm as tqdm
import time


def calculate_similarity(image_feature_local, text_feature_local):

    image_feature_local = image_feature_local.view(image_feature_local.size(0), -1)
    image_feature_local = image_feature_local / image_feature_local.norm(dim=1, keepdim=True)

    text_feature_local = text_feature_local.view(text_feature_local.size(0), -1)
    text_feature_local = text_feature_local / text_feature_local.norm(dim=1, keepdim=True)

    similarity = torch.mm(image_feature_local, text_feature_local.t())

    return similarity.cpu()


def calculate_ap(similarity, label_query, label_gallery):
    """
        calculate the similarity, and rank the distance, according to the distance, calculate the ap, cmc
    :param label_query: the id of query [1]
    :param label_gallery:the id of gallery [N]
    :return: ap, cmc
    """

    index = np.argsort(similarity)[::-1]  # the index of the similarity from huge to small
    good_index = np.argwhere(label_gallery == label_query)  # the index of the same label in gallery

    cmc = np.zeros(index.shape)

    mask = np.in1d(index, good_index)  # get the flag the if index[i] is in the good_index

    precision_result = np.argwhere(mask==True)  # get the situation of the good_index in the index

    precision_result = precision_result.reshape(precision_result.shape[0])

    if precision_result.shape[0] != 0:
        cmc[int(precision_result[0]):] = 1  # get the cmc

        d_recall = 1.0 / len(precision_result)
        ap = 0

        for i in range(len(precision_result)):  # ap is to calculate the PR area
            precision = (i + 1) * 1.0 / (precision_result[i] + 1)

            if precision_result[i] != 0:
                old_precision = i * 1.0 / precision_result[i]
            else:
                old_precision = 1.0

            ap += d_recall * (old_precision + precision) / 2

        return ap, cmc
    else:
        return None, None


def evaluate(similarity, label_query, label_gallery):
    label_query = label_query.numpy()
    label_gallery = label_gallery.numpy()

    cmc = np.zeros(label_gallery.shape)
    ap = 0
    for i in range(len(label_query)):
        ap_i, cmc_i = calculate_ap(similarity[i, :], label_query[i], label_gallery)
        cmc += cmc_i
        ap += ap_i
    """
    cmc_i is the vector [0,0,...1,1,..1], the first 1 is the first right prediction n,
    rank-n and the rank-k after it all add one right prediction, therefore all of them's index mark 1
    Through the  add all the vector and then divive the n_query, we can get the rank-k accuracy cmc
    cmc[k-1] is the rank-k accuracy   
    """
    cmc = cmc / len(label_query)
    map = ap / len(label_query)  # map = sum(ap) / n_query

    # print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (cmc[0], cmc[4], cmc[9], map))

    return cmc, map


def evaluate_without_matching_image(similarity, label_query, label_gallery, txt_img_index):
    label_query = label_query.numpy()
    label_gallery = label_gallery.numpy()

    cmc = np.zeros(label_gallery.shape[0] - 1)
    ap = 0
    count = 0
    for i in range(len(label_query)):

        similarity_i = similarity[i, :]
        similarity_i = np.delete(similarity_i, txt_img_index[i])
        label_gallery_i = np.delete(label_gallery, txt_img_index[i])
        ap_i, cmc_i = calculate_ap(similarity_i, label_query[i], label_gallery_i)
        if ap_i is not None:
            cmc += cmc_i
            ap += ap_i
        else:
            count += 1
    """
    cmc_i is the vector [0,0,...1,1,..1], the first 1 is the first right prediction n,
    rank-n and the rank-k after it all add one right prediction, therefore all of them's index mark 1
    Through the  add all the vector and then divive the n_query, we can get the rank-k accuracy cmc
    cmc[k-1] is the rank-k accuracy   
    """
    cmc = cmc / (len(label_query) - count)
    map = ap / (len(label_query) - count)  # map = sum(ap) / n_query
    # print('Rank@1:%f Rank@5:%f Rank@10:%f mAP:%f' % (cmc[0], cmc[4], cmc[9], map))

    return cmc, map


def load_checkpoint(model_root, model_name):
    filename = os.path.join(model_root, 'model', model_name)
    state = torch.load(filename, map_location='cpu')

    return state


def  write_result(similarity, img_labels, txt_labels):
  
    t2i_cmc, t2i_map = evaluate(similarity.T, txt_labels, img_labels)
    str2print = "t2i: @R1: {:.4}, @R5: {:.4}, @R10: {:.4}, map: {:.4}".format(t2i_cmc[0], t2i_cmc[4], t2i_cmc[9], t2i_map)
    print(str2print)
    return str2print 




def test(opt, epoch, network, img_dataloader, txt_dataloader, best, return_flag=True):

    best_txt_root = os.path.join(opt.save_path, 'log', 'best_test.log')
    txt_root = os.path.join(opt.save_path, 'log', 'test_separate.log')

    str = "Testing Epoch: {}".format(epoch)
    write_txt(str, txt_root)
    print(str)

    image_feature_global = []
    image_feature_local = []
    img_labels = []

    with h5py.File("images_data.h5", "w") as h5_file:
        for times, [image, label, image_path] in enumerate(tqdm(img_dataloader,desc="Processing images..")):

            h5_file.create_dataset(f"image_{times}", data=image, compression="gzip")
            h5_file.create_dataset(f"label_{times}", data=label, compression="gzip")

            # print("Image:{} label:{}".format(image.shape, label.shape))
            # print("Image:{} label:{}".format(image.dtype, label.dtype))
        

            image = image.to(opt.device)
            label = label.to(opt.device)


            print("Image shape is:{}".format(image.shape))
            with torch.no_grad():
                start_time = time.time()
                img_global_i, img_local_i = network.img_embedding(image)
                end_time = time.time()
                duration = end_time - start_time
                print(f"Single text process {duration} seconds.")

            image_feature_global.append(img_global_i)
            #image_feature_local.append(img_local_i)
            img_labels.append(label.view(-1))

        #image_feature_local = torch.cat(image_feature_local, 0)
        image_feature_global = torch.cat(image_feature_global, 0)
        img_labels = torch.cat(img_labels, 0)

        text_feature_local = []
        text_feature_global = []
        txt_labels = []

    with h5py.File("text_data.h5", "w") as h5_file:
        for times, [label, caption_code, caption_length, caption_mask] in enumerate(tqdm(txt_dataloader,desc="processing texts...")):
            
            # h5_file.create_dataset(f"label_{times}", data=label, compression="gzip")
            # h5_file.create_dataset(f"caption_code_{times}", data=caption_code, compression="gzip")
            # h5_file.create_dataset(f"caption_length_{times}", data=caption_length, compression="gzip")
            # # print("Label:{} caption_code:{} caption_length:{}".format(label.shape, caption_code.shape, caption_length.shape))
            # # print("Label:{} caption_code:{} caption_length:{}".format(label.dtype, caption_code.dtype, caption_length.dtype))
            # # break
            
            label = label.to(opt.device)
            caption_code = caption_code.to(opt.device).long()
            caption_length = caption_length.to(opt.device)

            with torch.no_grad():
                text_global_i, text_local_i = network.txt_embedding(caption_code, caption_length)

            #text_feature_local.append(text_local_i)
            text_feature_global.append(text_global_i)
            txt_labels.append(label.view(-1))
    
    
    #text_feature_local = torch.cat(text_feature_local, 0)
    text_feature_global = torch.cat(text_feature_global, 0)
    txt_labels = torch.cat(txt_labels, 0)

    #similarity_local = calculate_similarity(image_feature_local, text_feature_local).numpy()
    similarity_global = calculate_similarity(image_feature_global, text_feature_global).numpy()

    similarity_all = similarity_global #+ similarity_local

    img_labels = img_labels.cpu()
    txt_labels = txt_labels.cpu()

    best = write_result(similarity_global, img_labels, txt_labels, 'similarity_global:',
                        txt_root, best_txt_root, epoch, best)

    # best = write_result(similarity_local, img_labels, txt_labels, 'similarity_local:',
    #                     txt_root, best_txt_root, epoch, best)

    best = write_result(similarity_all, img_labels, txt_labels, 'similarity_all:',
                        txt_root, best_txt_root, epoch, best)

    if return_flag:
        return best