# coding=utf-8
import _ext.nms as nms_c
import numpy as np
import torch


# bbox_number 是一个batch_size 大小的一维numpy，其中的每个元素是一张图片的bbox数量
# 这个函数生成一个batch_size*max(bbox_number) 大小的LongTensor 并返回该tensor
# 例如:bbox_number=[2,3,1]
# 则生成的mask为
# [[1,1,0],[1,1,1],[1,0,0]]
# 其中1代表该位置的bbox有效 0代表无效
def _get_mask(bbox_number):
    max_bbox_number = bbox_number.max()
    mask = np.ones([np.size(bbox_number, 0), max_bbox_number], np.int64)
    for i in range(0, np.size(bbox_number, 0)):
        mask[i, bbox_number[i]:-1] = 0
    return torch.Tensor(mask).cuda()


# bbox [batch_size,max_bbox_number,4] long cuda tensor
# bbox_number [batch_size] 是一个batch_size 大小的一维numpy，其中的每个元素是一张图片的bbox数量
# score [batch_size,max_bbox_number] float cuda tensor
# thresh float
# 返回按照score降序排列的bbox,score和有效位的mask
def nms(bbox, bbox_number, score, thresh):
    if(bbox.size()[0] != np.size(bbox_number, 0)):
        raise RuntimeError('the bbox size must equal the bbox_number size')
    if(bbox.size()[0] != score.size()[0] or bbox.size()[1] != score.size()[1]):
        raise RuntimeError('the bbox size must equal the score size')
    mask = _get_mask(bbox_number)
    score = mask*score
    score, idx = torch.sort(score, dim=1, descending=True)
    for i in range(0, bbox.size()[0]):
        bbox[i, :, :] = bbox[i, idx[i], :]
    nms_c.nms(bbox, mask, thresh)
    return bbox, score, mask
