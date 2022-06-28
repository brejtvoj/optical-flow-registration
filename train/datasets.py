import numpy as np
import random
import torch
import torch.utils.data as data
from torchvision.transforms import ColorJitter
import os
import cv2
from PIL import Image


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=True):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.5
        self.stretch_prob = 0.5
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5/3.14)
        self.asymmetric_color_aug_prob = 0.1
        self.eraser_aug_prob = 0.5

    def color_transform(self, img1, img2):
        """ Photometric augmentation """
        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            img1 = np.array(self.photo_aug(Image.fromarray(img1)), dtype=np.uint8)
            img2 = np.array(self.photo_aug(Image.fromarray(img2)), dtype=np.uint8)

        # symmetric
        else:
            print(img1.shape, img2.shape)
            image_stack = np.concatenate([img1, img2], axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            img1, img2 = np.split(image_stack, 2, axis=0)

        return img1, img2

    def eraser_transform(self, img1, img2):
        """ Occlusion augmentation """
        bounds_l = 0.05 * (img1.shape[0] + img1.shape[1]) / 2.
        bound_u = 4 * bounds_l
        bounds = [int(bounds_l), int(bound_u)]

        ht, wd = img1.shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            mean_color = np.mean(img2.reshape(-1, 3), axis=0)
            for _ in range(np.random.randint(1, 3)):
                x0 = np.random.randint(0, wd)
                y0 = np.random.randint(0, ht)
                dx = np.random.randint(bounds[0], bounds[1])
                dy = np.random.randint(bounds[0], bounds[1])
                img2[y0:y0+dy, x0:x0+dx, :] = mean_color

        return img1, img2

    def spatial_transform(self, img1, img2, flow, mask=None):
        # randomly sample scale
        ht, wd = img1.shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)

        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            img1 = cv2.resize(img1, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            img2 = cv2.resize(img2, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            if mask is not None:
                mask = cv2.resize(mask, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR)
            flow = flow * [scale_x, scale_y]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob: # h-flip
                img1 = img1[:, ::-1]
                img2 = img2[:, ::-1]
                flow = flow[:, ::-1] * [-1.0, 1.0]
                if mask is not None:
                    mask = mask[:, ::-1]

            if np.random.rand() < self.v_flip_prob: # v-flip
                img1 = img1[::-1, :]
                img2 = img2[::-1, :]
                flow = flow[::-1, :] * [1.0, -1.0]
                if mask is not None:
                    mask = mask[::-1, :]


        # Random crop
        y0 = np.random.randint(0, img1.shape[0] - self.crop_size[0])
        x0 = np.random.randint(0, img1.shape[1] - self.crop_size[1])

        img1 = img1[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        img2 = img2[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        flow = flow[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]
        if mask is not None:
            mask = mask[y0:y0+self.crop_size[0], x0:x0+self.crop_size[1]]

        if mask is not None:
            return img1, img2, flow, mask
        else:
            return img1, img2, flow, None

    def __call__(self, img1, img2, flow, mask=None):
        img1, img2 = self.color_transform(img1, img2)
        for i in range(5):
            img1, img2 = self.eraser_transform(img1, img2)
        for i in range(5):
            img2, img1 = self.eraser_transform(img2, img1)
        img1, img2, flow, mask = self.spatial_transform(img1, img2, flow, mask)

        img1 = np.ascontiguousarray(img1)
        img2 = np.ascontiguousarray(img2)
        flow = np.ascontiguousarray(flow)
        if mask is not None:
            mask = np.ascontiguousarray(mask)

        return img1, img2, flow, mask



class ANHIR(data.Dataset):
    def __init__(self, aug_params, root, sparse=False, count=500,
                 use_mask=False):
        self.flow_list = self.load_flows(root, count)
        self.image_list = self.load_images(root, count, mode='original')
        self.image_t_list = self.load_images(root, count, mode='target')
        self.use_mask = use_mask
        if self.use_mask:
            self.mask_list = self.load_masks(root, count)

        self.init_seed = False
        self.augmentor = FlowAugmentor(**aug_params)

    def __getitem__(self, index):
        if not self.init_seed:
             worker_info = torch.utils.data.get_worker_info()
             if worker_info is not None:
                 torch.manual_seed(worker_info.id)
                 np.random.seed(worker_info.id)
                 random.seed(worker_info.id)
                 self.init_seed = True

        index = index % len(self.image_list)
        valid = None

        flow = self.read_flow_file(self.flow_list[index])
        img1 = self.read_image_file(self.image_list[index])
        img2 = self.read_image_file(self.image_t_list[index])
        if self.use_mask:
            mask = self.read_mask_file(self.mask_list[index])
        else:
            mask = None

        flow = np.array(flow).astype(np.float32)
        img1 = np.array(img1).astype(np.uint8)
        img2 = np.array(img2).astype(np.uint8)

        # grayscale images
        if len(img1.shape) == 2:
            img1 = np.tile(img1[...,None], (1, 1, 3))
            img2 = np.tile(img2[...,None], (1, 1, 3))
        else:
            img1 = img1[..., :3]
            img2 = img2[..., :3]
        if self.augmentor is not None:
            if self.sparse:
                img1, img2, flow, valid = self.augmentor(img1, img2, flow, valid)
            else:
                img1, img2, flow, mask = self.augmentor(img1, img2, flow, mask)

        img1 = torch.from_numpy(img1).permute(2, 0, 1).float()
        img2 = torch.from_numpy(img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()
        if self.use_mask:
            mask = torch.from_numpy(mask).float()

        if valid is not None:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1000) & (flow[1].abs() < 1000)
        if self.use_mask:
            return img1, img2, flow, valid.float(), mask
        else:
            return img1, img2, flow, valid.float()


    def __rmul__(self, v):
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        return self
         
    def __len__(self):
        return len(self.image_list)

    def load_images(self, path, count, mode='original'):
        images = list()
        for image in os.listdir(path):

            if mode == 'original':
                if 't' in image or '.flo' in image:
                    continue
            else:
                if 't' not in image:
                    continue
            image_path = os.path.join(path, image).replace('\\', '/')
            images.append(image_path)
            if len(images) == count:
                break
        return images

    def load_flows(self, path, count):
        flows = list()
        for flow in os.listdir(path):
            if not '.flo' in flow:
                continue
            flow_path = os.path.join(path, flow).replace('\\', '/')
            flows.append(flow_path)
            if len(flows) == count:
                break
        return flows

    def load_masks(self, path, count):
        masks = list()
        for mask in os.listdir(path):
            if not 'mask' in mask:
                continue
            mask_path = os.path.join(path, mask).replace('\\', '/')
            masks.append(mask_path)
            if len(masks) == count:
                break
        return masks

    def read_image_file(self, file_name):
        return np.array(Image.open(file_name))

    def read_flow_file(self, file_name):
        with open(file_name, "rb") as f:
            magic = np.fromfile(f, "c", count=4).tobytes()
            if magic != b"PIEH":
                raise ValueError("Magic number incorrect. Invalid .flo file")

            w = int(np.fromfile(f, "<i4", count=1))
            h = int(np.fromfile(f, "<i4", count=1))
            data = np.fromfile(f, "<f4", count=2 * w * h)
            return data.reshape(h, w, 2)

    def read_mask_file(self, file_name):
        return np.load(file_name, allow_pickle=True)

def fetch_dataloader(args, root, count):
    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.8, 'do_flip': True}
    train_dataset = ANHIR(aug_params, root, count, use_mask=False)

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
        pin_memory=False, shuffle=True, num_workers=1, drop_last=True)
    print('Training with %d image pairs' % len(train_dataset))
    return train_loader