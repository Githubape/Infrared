import os
import os.path as osp
import torch
import torch.utils.data as data
import cv2
import numpy as np
import yaml
from sklearn.metrics import confusion_matrix


def tensor_to_image(torch_tensor, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    """
    Converts a 3D Pytorch tensor into a numpy array for display

    Parameters:
        torch_tensor -- Pytorch tensor in format(channels, height, width)
    """
    for t, m, s in zip(torch_tensor, mean, std):
        t.mul_(s).add_(m)

    return np.uint8(torch_tensor.mul(255.0).numpy().transpose(1, 2, 0))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Metrics:
    """
    Calculates all the metrics reported in paper: Overall Accuracy, Average Accuracy,
    mean IOU and mean DICE score
    Ref: https://github.com/rmkemker/EarthMapper/blob/master/metrics.py

    Parameters:
        ignore_index -- which particular index to ignore when calculating all values.
                        In AeroRIT, index '5' is the undefined class and hence, the
                        default value for this function.
    """

    def __init__(self, ignore_index=999):
        self.ignore_index = ignore_index

    def __call__(self, truth, prediction):

        self.truth = truth.flatten()
        self.prediction = prediction.flatten()

        ignore_loc = np.where(self.truth == self.ignore_index)
        self.truth = np.delete(self.truth, ignore_loc)
        self.prediction = np.delete(self.prediction, ignore_loc)

        self.c = confusion_matrix(self.truth, self.prediction)
        # self.mb_iou = self._m_b_iou(truth, prediction)
        return self._oa(), self._aa(), self._mIOU(), self._dice_coefficient(), self._IOU(), self.c

    def _oa(self):
        return np.sum(np.diag(self.c)) / np.sum(self.c)

    def _aa(self):
        return np.nanmean(np.diag(self.c) / (np.sum(self.c, axis=1) + 1e-10))

    def _IOU(self):
        intersection = np.diag(self.c)
        ground_truth_set = self.c.sum(axis=1)
        predicted_set = self.c.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection + 1e-10

        intersection_over_union = intersection / union.astype(np.float32)

        return intersection_over_union

    def _mIOU(self):
        intersection_over_union = self._IOU()
        return np.nanmean(intersection_over_union)

    def _dice_coefficient(self):
        intersection = np.diag(self.c)
        ground_truth_set = self.c.sum(axis=1)
        predicted_set = self.c.sum(axis=0)
        dice = (2 * intersection) / (ground_truth_set + predicted_set + 1e-10)
        avg_dice = np.nanmean(dice)
        return avg_dice

    # General util function to get the boundary of a binary mask.
    def _mask_to_boundary(self, mask, dilation_ratio=0.02):
        """
        Convert binary mask to boundary mask.
        :param mask (numpy array, uint8): binary mask
        :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        :return: boundary mask (numpy array)
        """
        h, w = mask.shape
        img_diag = np.sqrt(h ** 2 + w ** 2)
        dilation = int(round(dilation_ratio * img_diag))
        if dilation < 1:
            dilation = 1
        # Pad image so mask truncated by the image border is also considered as boundary.
        new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        kernel = np.ones((3, 3), dtype=np.uint8)
        new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
        mask_erode = new_mask_erode[1: h + 1, 1: w + 1]
        # G_d intersects G in the paper.
        return mask - mask_erode

    def _boundary_iou(self, gt, dt, dilation_ratio=0.02):
        """
        Compute boundary iou between two binary masks.
        :param gt (numpy array, uint8): binary mask
        :param dt (numpy array, uint8): binary mask
        :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
        :return: boundary iou (float)
        """
        gt_boundary = self._mask_to_boundary(gt, dilation_ratio)
        dt_boundary = self._mask_to_boundary(dt, dilation_ratio)
        intersection = ((gt_boundary * dt_boundary) > 0).sum()
        union = ((gt_boundary + dt_boundary) > 0).sum()
        boundary_iou = intersection / union
        return boundary_iou

    def _m_b_iou(self, truth, prediction, dilation_ratio=0.02):
        mb_iou = []

        for i in range(truth.shape[0]):
            mb_iou += self._boundary_iou(truth[i], prediction[i])

        return mb_iou


class AeroLoader(data.Dataset):
    """
    This function serves as the dataloader for the AeroCampus dataset

    Parameters:
        set_type    -- 'train' or 'test'
        transforms  -- transforms for RGB image (default: normalize)
        augs        -- augmentations used (default: horizontal & vertical image flips)
    """

    def __init__(self, set_type='train', transforms=None, augs=None):

        self.working_dir = osp.join('dataset/aeroscapes')

        self.rgb_dir = 'JPEGImages'
        self.label_dir = 'SegmentationClass'

        self.transforms = transforms
        self.augmentations = augs

        self.n_classes = 12

        with open(osp.join(self.working_dir, 'ImageSets/' + set_type + '.txt')) as f:
            self.filelist = f.read().splitlines()

    def __getitem__(self, index):
        rgb = cv2.imread(osp.join(self.working_dir, self.rgb_dir, self.filelist[index] + '.jpg'))
        rgb = rgb[:, :, ::-1]

        label = cv2.imread(osp.join(self.working_dir, self.label_dir, self.filelist[index] + '.png'), 0)

        if self.augmentations is not None:
            rgb, label = self.augmentations(rgb, label)

        if self.transforms is not None:
            rgb = self.transforms(rgb)

            # label = self.encode_segmap(label)
            label = torch.from_numpy(np.array(label)).long()

        # return hsi, label
        return rgb, label

    def __len__(self):
        return len(self.filelist)

    # def get_labels(self):
    #     return np.asarray(
    #         [
    #             [192, 128, 128],
    #             [0, 128, 0],
    #             [128, 128, 128],
    #             [128, 0, 0],
    #             [0, 0, 128],
    #             [192, 0, 128],
    #             [192, 0, 0],
    #             [192, 128, 0],
    #             [0, 64, 0],
    #             [128, 128, 0],
    #             [0, 128, 128],
    #         ]
    #     )
    #
    # def encode_segmap(self, mask):
    #     mask = mask.astype(int)
    #     label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    #     for ii, label in enumerate(self.get_labels()):
    #         label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    #     label_mask = label_mask.astype(int)
    #     return label_mask
    #
    # def decode_segmap(self, label_mask, plot=False):
    #     label_colours = self.get_labels()
    #     r = label_mask.copy()
    #     g = label_mask.copy()
    #     b = label_mask.copy()
    #     for ll in range(0, self.n_classes):
    #         r[label_mask == ll] = label_colours[ll, 0]
    #         g[label_mask == ll] = label_colours[ll, 1]
    #         b[label_mask == ll] = label_colours[ll, 2]
    #     rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    #     rgb[:, :, 0] = r
    #     rgb[:, :, 1] = g
    #     rgb[:, :, 2] = b
    #
    #     return np.uint8(rgb)


def parse_args(parser):
    """
    Standard argument parser
    """
    args = parser.parse_args()
    if args.config_file and os.path.exists(args.config_file):
        data = yaml.safe_load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        #        print (data)
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    return args


def pred_to_rgb(label_mask, label_colours, n_classes):
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    return np.uint8(rgb)



