import os
import torch
from torch.utils.data import Dataset
import SimpleITK as sitk
import torchio as tio
from config import config
from torchio.data import UniformSampler
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
    OneHot,
       Resize
)


def Tooth_Dataset(images_dir, labels_dir, train=True):
    subjects_list = []
    images_list = os.listdir(images_dir)

    labels_binary_dir = os.path.join(labels_dir, 'binary')
    labels_tooth_dir = os.path.join(labels_dir, 'tooth')
    labels_bone_dir = os.path.join(labels_dir, 'bone')

    labels_binary_list = os.listdir(labels_binary_dir)
    labels_tooth_list = os.listdir(labels_tooth_dir)
    labels_bone_list = os.listdir(labels_bone_dir)

    # queue_length = config.queue_length
    # samples_per_volume = config.samples_per_volume
    # patch_size = config.patch_size

    training_transform = Compose([
        RandomFlip(),
        RandomNoise(),
        RandomMotion(),
	Resize(target_shape=64)
    ])
    for image, labels_binary, labels_tooth, labels_bone in zip(images_list, labels_binary_list, labels_tooth_list, labels_bone_list):
        subject = tio.Subject(
            image=tio.ScalarImage(os.path.join(images_dir, image)),
            labels_binary=tio.LabelMap(os.path.join(labels_binary_dir, labels_binary)),
            labels_tooth=tio.LabelMap(os.path.join(labels_tooth_dir, labels_tooth)),
            labels_bone=tio.LabelMap(os.path.join(labels_bone_dir, labels_bone)),
        )
        subjects_list.append(subject)

    if train:
        subject_dataset = tio.SubjectsDataset(subjects_list, transform=training_transform)
        # queue_dataset = tio.Queue(
        #     subject_dataset,
        #     max_length=queue_length,
        #     samples_per_volume=samples_per_volume,
        #     sampler=tio.LabelSampler(patch_size=patch_size, label_name=None,
        #                              label_probabilities={0: 0, 1: 5, 2: 5, 3: 2}),
        # )

        return subject_dataset

    else:
        subject_dataset = tio.SubjectsDataset(subjects_list, transform=Resize(target_shape=64))
        return subject_dataset


