import nms
import numpy as np
import torch


def test_nms():
    bbox = [[
        [0, 0, 6, 6], [1, 1, 7, 7], [5, 5, 12, 12]],
        [[0, 0, 6, 6], [0, 1, 6, 7], [0, 0, 0, 0]]]
    bbox = torch.LongTensor(bbox).cuda()

    score = [
        [0.2, 0.8, 0.5],
        [0.5, 0.4, 0.0]]
    score = torch.Tensor(score).cuda()

    bbox_number = [3, 2]
    bbox_number = np.array(bbox_number)
    thresh = 0.5

    target_bbox = [[
        [1, 1, 7, 7], [5, 5, 12, 12], [0, 0, 6, 6]],
        [[0, 0, 6, 6], [0, 1, 6, 7], [0, 0, 0, 0]]
    ]
    target_bbox = torch.LongTensor(target_bbox).cuda()

    target_score = [
        [0.8, 0.5, 0.2],
        [0.5, 0.4, 0.0]]
    target_score = torch.Tensor(target_score).cuda()

    target_mask = [
        [1, 0, 1],
        [1, 0, 0]
    ]
    target_mask = torch.Tensor(target_mask).cuda()

    sort_bbox, sort_score, mask = nms.nms(bbox, bbox_number, score, thresh)

    if(torch.sum((target_bbox-sort_bbox)) == 0):
        print('test succeed in sort_bbox')
    else:
        print('test fail in sort_bbox')
    if(torch.sum((target_score-sort_score)) == 0):
        print('test succeed in sort_score')
    else:
        print('test fail in sort_score')
    if(torch.sum((mask-target_mask)) == 0):
        print('test succeed in mask')
    else:
        print('test fail in mask')


if __name__ == '__main__':
    test_nms()
