import numpy as np
import os

from data_management import save_flow, save_image

from generate_flow_from_landmarks import generate_training_data

from generate_flow_affine_and_synthetic import FlowGenerator as FlowGenAffine


class DatasetGenerator():
    """Dataset for generation and saving of the training triplets and mask."""

    def __init__(self, starting_idx=1):
        self.images = list()
        self.idx = starting_idx

    def generate_names(self):
        """Generate names for files."""
        idx = str(self.idx).zfill(4)
        self.idx += 1

        img = 'img_' + idx
        img_t = 'img_t_' + idx
        flow = 'flow_' + idx
        mask = 'mask_' + idx

        return img, img_t, flow, mask

    def generate(self, path, save_path, variant='landmarks',
                 image_count=2000, create_mask=False):
        """
        Generate the training triplet and the mask and save them.

        Parameters
        ----------
        path : string
            Location of the original dataset.
        save_path : string
            Location of the save folder
        variant : string
            Triplet generation method ('landmarks' / 'affine_synth')
        image_count : int
            Number of triplets to be generated.
        create_mask : bool
            Determin, whether to create the mask.
        """
        def gen_data(path, idx, create_mask):
            if variant == 'landmarks':
                data_blob = generate_training_data(path, idx, create_mask)
            elif variant == 'affine_synth':
                gen = FlowGenAffine(path)
                data_blob = gen(use_mask=create_mask)
            return data_blob

        for i in range(image_count):
            # Name generation
            img_name, img_t_name, flow_name, mask_name = self.generate_names()
            img_name = os.path.join(save_path, img_name).replace('\\', '/') + '.png'
            img_t_name = os.path.join(save_path, img_t_name).replace('\\', '/') + '.png'
            flow_name = os.path.join(save_path, flow_name).replace('\\', '/') + '.flo'
            mask_name = os.path.join(save_path, mask_name).replace('\\', '/') + '.npy'

            # Create data
            try:
                data = gen_data(path, self.idx, create_mask)
                image1, image2, flow, mask, fail = data
            except FileNotFoundError:
                print('Loading error occured...')
                self.idx -= 1
                i -= 1
                continue

            # File saving
            flow = np.transpose((flow), (1, 2, 0))
            save_flow(flow, flow_name)
            save_image(image1, img_name, const=1)
            save_image(image2, img_t_name, const=1)

            print(f'IDX: {self.idx - 1} | Shape: {image1.shape}')
