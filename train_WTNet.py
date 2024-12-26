import torchio as tio
import os
import argparse
from shutil import copy
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from modles.WTNet import WTNet
from WTNet.config import config
from WTNet.dataloader import Tooth_Dataset
from torch.nn.functional import softmax
from monai.losses.dice import DiceLoss
from WTNet.train_and_eval import train_one_epoch, eval, save_checkpoint

def create_model():
    model = WTNet()
    return model


def main(args, fold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_workers = args.num_workers
    best_val_loss = 1000
    best_epoch = 0
    count = 0
    fold = fold

    train_dataset = Tooth_Dataset(images_dir=args.input_image_dir_train, labels_dir=args.label_dir_train, train=True)
    val_dataset = Tooth_Dataset(images_dir=args.input_image_dir_val, labels_dir=args.label_dir_val, train=False)
    NKUT_Train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    NKUT_Val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    if args.resume:
        model = create_model()
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=0.8,
                                                               patience=3,
                                                               verbose=True,
                                                               )

        checkpoint = torch.load('/data/dataset/zzh/ckeckpoint/fold{}/checkpoint_epoch70.pth'.format(fold))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optim'])
        ckpt_epoch = checkpoint['epoch']
        scheduler.load_state_dict(checkpoint['scheduler'])

    else:
        model = create_model()
        model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                               mode='min',
                                                               factor=0.8,
                                                               patience=3,
                                                               verbose=True)
        ckpt_epoch = args.start_epoch

    writer = SummaryWriter(log_dir=args.log_dir)
    diceloss = DiceLoss(to_onehot_y=True, softmax=True)
    celoss = CrossEntropyLoss()

    for epoch in range(ckpt_epoch, args.epochs + 1):
        loss_result, new_model = train_one_epoch(model=model, diceloss=diceloss, celoss=celoss,
                                                 optimizer=optimizer,
                                                 dataloader=NKUT_Train_loader, device=device, arg=args, epoch=epoch)
        writer.add_scalar('Training Loss', loss_result, epoch)

        save_checkpoint(model=new_model, optim=optimizer, scheduler=scheduler, checkpoint_dir=args.ckpt_dir,
                        epoch=epoch, save_fre=args.epochs_per_checkpoint)

        val_loss_sum = eval(model_path='./result/WTNet/fold{}/latest_output.pth'.format(fold),
                            dataloader=NKUT_Val_loader, device=device, diceloss=diceloss,
                            celoss=celoss)

        scheduler.step(loss_result)
        writer.add_scalar('Val Loss', val_loss_sum, epoch)

        if best_val_loss > val_loss_sum:
            copy(src=args.latest_output_dir, dst=args.best_model_path)
            best_val_loss = val_loss_sum
            best_epoch = epoch
            count = 0
        else:
            count += 1

        print('The total val loss is {}, best is {}, in Epoch {}'.format(val_loss_sum, best_val_loss, best_epoch))
        model = new_model

        if count == args.early_stop:
            print("early stop")
            break

def parse_args(fold):

    parser = argparse.ArgumentParser(description='NKUT Wisdom Tooth Segmentation')
    parser.add_argument('-epochs', type=int, default=500, help='Numbers of epochs to train')
    parser.add_argument('-batch_size', type=int, default=3, help='batch size')
    parser.add_argument('-input_image_dir_train', type=str, default='/data/dataset/zzh/NKUT/patch/64_64_64/fold{}/Train/Image'.format(fold))
    parser.add_argument('-label_dir_train', type=str, default='/data/dataset/zzh/NKUT/patch/64_64_64/fold{}/Train/Label'.format(fold))
    parser.add_argument('-input_image_dir_val', type=str, default='/data/dataset/zzh/NKUT/patch/64_64_64/fold4/Val/Image')
    parser.add_argument('-label_dir_val', type=str, default='/data/dataset/zzh/NKUT/patch/64_64_64/fold4/Val/Label')

    parser.add_argument('-epochs-per-checkpoint', type=int, default=5, help='Number of epochs per checkpoint')
    parser.add_argument('-log_dir', '-output_logs_dir', type=str, default='./logs', help='Where to save the train logs')
    parser.add_argument('-lr', '-learning rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('-latest_output_dir', type=str, default='./result/WTNet/fold{}/latest_output.pth'.format(fold),
                        help='where to store the latest model')
    parser.add_argument('-best_model_path', type=str, default='./result/WTNet/fold{}/best_result.pth'.format(fold),
                        help='where to save the best val model')
    parser.add_argument('-ckpt_dir', type=str, default='/data/dataset/zzh/ckeckpoint/fold{}'.format(fold),
                        help='where to save the latest checkpoint')
    parser.add_argument('-epochs_per_checkpoint', type=int, default=5, help='epoch to store a checkpoint')
    parser.add_argument('-resume', action='store_true', help='continue training')
    parser.add_argument('-early_stop', type=int, default=200, help='early stop')
    parser.add_argument('-num_workers', type=int, default=8, help='num_workers')
    parser.add_argument('-start_epoch', type=int, default=1, help='num_workers')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    fold = 3
    args = parse_args(fold)
    main(args, fold)
