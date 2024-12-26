import torch
from torch import nn
from tqdm import tqdm
import os
import torchio as tio


def train_one_epoch(model, diceloss, celoss, optimizer, dataloader, device, epoch, arg):
    model.train()
    dice_loss = diceloss
    ce_loss = celoss

    loss_sum = 0
    iteration = 0

    with tqdm(enumerate(dataloader), total=len(dataloader)) as loop:
        for i, batch in loop:
            data = batch['image'][tio.DATA]
            labels_binary = batch['labels_binary'][tio.DATA]
            labels_tooth = batch['labels_tooth'][tio.DATA]
            labels_bone = batch['labels_bone'][tio.DATA]

            data = data.float()
            labels_binary = labels_binary.long()
            labels_tooth = labels_tooth.long()
            labels_bone = labels_bone.long()

            data = torch.transpose(data, 2, 4)
            labels_binary = torch.transpose(labels_binary, 2, 4)
            labels_tooth = torch.transpose(labels_tooth, 2, 4)
            labels_bone = torch.transpose(labels_bone, 2, 4)

            data = data.to(device)
            labels_binary = labels_binary.to(device)
            labels_tooth = labels_tooth.to(device)
            labels_bone = labels_bone.to(device)

            Binary_out, out_tooth_last, out_bone_last = model(data)
            Dice_Binary = dice_loss(Binary_out, labels_binary)
            Dice_tooth = dice_loss(out_tooth_last, labels_tooth)
            Dice_bone = dice_loss(out_bone_last, labels_bone)

            CE_Binary = ce_loss(Binary_out, labels_binary.squeeze(1))
            CE_tooth = ce_loss(out_tooth_last, labels_tooth.squeeze(1))
            CE_bone = ce_loss(out_bone_last, labels_bone.squeeze(1))

            loss = Dice_Binary+CE_Binary+Dice_tooth+CE_tooth+Dice_bone+CE_bone

            loss_sum += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iteration += 1

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(lr=optimizer.state_dict()['param_groups'][0]['lr'], total_loss=loss_sum / iteration)

    torch.save(model, arg.latest_output_dir)

    return loss_sum / iteration, model


def eval(model_path, dataloader, device, diceloss, celoss):
    model = torch.load(model_path)
    model.to(device)
    model.eval()
    iteration = 0

    dice_loss = diceloss.to(device)
    ce_loss = celoss.to(device)
    val_loss_sum = 0

    with torch.no_grad():
        with tqdm(enumerate(dataloader)) as loop_val:
            for i, batch in loop_val:
                data = batch['image'][tio.DATA]
                labels_binary = batch['labels_binary'][tio.DATA]
                labels_tooth = batch['labels_tooth'][tio.DATA]
                labels_bone = batch['labels_bone'][tio.DATA]

                data = data.float()
                labels_binary = labels_binary.long()
                labels_tooth = labels_tooth.long()
                labels_bone = labels_bone.long()

                data = torch.transpose(data, 2, 4)
                labels_binary = torch.transpose(labels_binary, 2, 4)
                labels_tooth = torch.transpose(labels_tooth, 2, 4)
                labels_bone = torch.transpose(labels_bone, 2, 4)

                data = data.to(device)
                labels_binary = labels_binary.to(device)
                labels_tooth = labels_tooth.to(device)
                labels_bone = labels_bone.to(device)

                Binary_out, out_tooth_last, out_bone_last = model(data)
                Dice_Binary = dice_loss(Binary_out, labels_binary)
                Dice_tooth = dice_loss(out_tooth_last, labels_tooth)
                Dice_bone = dice_loss(out_bone_last, labels_bone)

                CE_Binary = ce_loss(Binary_out, labels_binary.squeeze(1))
                CE_tooth = ce_loss(out_tooth_last, labels_tooth.squeeze(1))
                CE_bone = ce_loss(out_bone_last, labels_bone.squeeze(1))

                loss = Dice_Binary + CE_Binary + Dice_tooth + CE_tooth + Dice_bone + CE_bone

                val_loss_sum += loss.item()
                iteration += 1

    return val_loss_sum / iteration


def save_checkpoint(model, optim, scheduler, epoch, save_fre, checkpoint_dir):
    if epoch % save_fre == 0:
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optim.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
            },
            os.path.join(checkpoint_dir, 'checkpoint_epoch{}.pth'.format(epoch))
        )

















