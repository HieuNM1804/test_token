import os
import glob
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image

SKETCHY_2_UNSEEN_CLASSES = [
    "bat",
    "cabin",
    "cow",
    "dolphin",
    "door",
    "giraffe",
    "helicopter",
    "mouse",
    "pear",
    "raccoon",
    "rhinoceros",
    "saw",
    "scissors",
    "seagull",
    "skyscraper",
    "songbird",
    "sword",
    "tree",
    "wheelchair",
    "windmill",
    "window",
]

class Sketchy(torch.utils.data.Dataset):
    # Training dataset
    def __init__(self, opts, transform, mode='train', return_orig=False):

        self.opts = opts
        self.transform = transform
        self.return_orig = return_orig

        self.all_categories = self.categories_for_mode(opts, mode)

        self.all_sketches_path = []
        self.all_photos_path = {}

        for category in self.all_categories:
            self.all_sketches_path.extend(
                sorted(glob.glob(os.path.join(self.opts.data_dir, 'sketch', category, '*.png'))))
            self.all_photos_path[category] = sorted(
                glob.glob(os.path.join(self.opts.data_dir, 'photo', category, '*.jpg')))

            if len(self.all_photos_path[category]) == 0:
                raise RuntimeError('No photos found for category: %s' % category)

        if len(self.all_sketches_path) == 0:
            raise RuntimeError('No sketches found under: %s' % self.opts.data_dir)

    def __len__(self):
        return len(self.all_sketches_path)
        
    def __getitem__(self, index):
        filepath = self.all_sketches_path[index]                
        category = filepath.split(os.path.sep)[-2]
        filename = os.path.basename(filepath)
        
        neg_classes = self.all_categories.copy()
        neg_classes.remove(category)

        sk_path  = filepath
        img_path = np.random.choice(self.all_photos_path[category])
        neg_path = np.random.choice(self.all_photos_path[np.random.choice(neg_classes)])

        sk_data  = Image.open(sk_path).convert('RGB')
        img_data = Image.open(img_path).convert('RGB')
        neg_data = Image.open(neg_path).convert('RGB')

        sk_tensor  = self.transform(sk_data)
        img_tensor = self.transform(img_data)
        neg_tensor = self.transform(neg_data)
        
        if self.return_orig:
            return (sk_tensor, img_tensor, neg_tensor, category, filename,
                sk_data, img_data, neg_data)
        else:
            return (sk_tensor, img_tensor, neg_tensor, category, filename)

    @staticmethod
    def data_transform(opts):
        dataset_transforms = transforms.Compose([
            transforms.Resize(opts.max_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(opts.max_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711])
        ])
        return dataset_transforms

    @staticmethod
    def categories_for_mode(opts, mode='train'):
        if mode not in ('train', 'val'):
            raise ValueError("mode must be either 'train' or 'val'")

        sketch_root = os.path.join(opts.data_dir, 'sketch')
        if not os.path.isdir(sketch_root):
            raise RuntimeError('Sketch directory not found: %s' % sketch_root)

        all_categories = sorted([
            category for category in os.listdir(sketch_root)
            if category != '.ipynb_checkpoints'
            and os.path.isdir(os.path.join(sketch_root, category))
        ])

        missing = sorted(set(SKETCHY_2_UNSEEN_CLASSES) - set(all_categories))
        if missing:
            raise RuntimeError(
                'The Sketchy-2 split is missing categories: %s' % ', '.join(missing))

        if mode == 'train':
            return sorted(
                set(all_categories) - set(SKETCHY_2_UNSEEN_CLASSES))
        return list(SKETCHY_2_UNSEEN_CLASSES)


class SketchyRetrieval(torch.utils.data.Dataset):
    # Evaluation dataset
    def __init__(self, opts, transform, categories, modality):
        if modality not in ('sketch', 'photo'):
            raise ValueError("modality must be either 'sketch' or 'photo'")

        self.transform = transform
        self.modality = modality
        self.samples = []
        patterns = ('*.png',) if modality == 'sketch' else ('*.jpg',)

        for category_index, category in enumerate(categories):
            category_paths = []
            for pattern in patterns:
                category_paths.extend(glob.glob(
                    os.path.join(opts.data_dir, modality, category, pattern)))

            category_paths = sorted(set(category_paths))
            if not category_paths:
                raise RuntimeError(
                    'No %s files found for category: %s' % (modality, category))

            self.samples.extend([
                (path, category_index) for path in category_paths
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, category_index = self.samples[index]
        image = Image.open(path).convert('RGB')
        return self.transform(image), category_index, index


if __name__ == '__main__':
    from experiments.options import opts
    import tqdm

    dataset_transforms = Sketchy.data_transform(opts)
    dataset_train = Sketchy(opts, dataset_transforms, mode='train', return_orig=True)
    dataset_val = Sketchy(
        opts, dataset_transforms, mode='val', return_orig=True)

    idx = 0
    for data in tqdm.tqdm(dataset_val):
        continue
        (sk_tensor, img_tensor, neg_tensor, category, filename,
            sk_data, img_data, neg_data) = data

        canvas = Image.new('RGB', (224*3, 224))
        offset = 0
        for im in [sk_data, img_data, neg_data]:
            canvas.paste(im, (offset, 0))
            offset += im.size[0]
        canvas.save('output/%d.jpg'%idx)
        idx += 1
