import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
import albumentations
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

torch.backends.cudnn.benchmark = True

# model definition


class Discriminator(nn.Module):
    def __init__(self, in_channels=3) -> None:
        super().__init__()
        self.convlayers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=64,
                      kernel_size=4, stride=2, padding=1, padding_mode="reflect",),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                      stride=2, bias=False, padding_mode="reflect",),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4,
                      stride=2, bias=False, padding_mode="reflect",),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4,
                      stride=1, bias=False, padding_mode="reflect",),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4,
                      stride=1, padding=1, padding_mode="reflect",),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x, y) -> None:
        out = torch.cat([x, y], dim=1)
        out = self.convlayers(out)
        return out

# generator class definition


class Generator(nn.Module):
    def encoder(self, in_channels, out_channel, is_relu=False, need_batch_norm=True):
        x = nn.Sequential(
            nn.Conv2d(in_channels, out_channel, 4, 2, 1,
                      bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channel) if need_batch_norm else None,
            nn.ReLU() if is_relu else nn.LeakyReLU(),
        )
        return x

    def decoder(self, in_channels, out_channel, is_relu=False, need_batch_norm=True, need_dropout=True, ):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU() if is_relu else nn.LeakyReLU(),
        ) if not need_dropout else nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channel, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU() if is_relu else nn.LeakyReLU(),
            nn.Dropout(0.5),
        )

    def __init__(self, in_channels=3, features=64):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4,
                      stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )

        self.layer2 = self.encoder(
            in_channels=features, out_channel=features * 2)
        self.layer3 = self.encoder(features * 2, features * 4)
        self.layer4 = self.encoder(features * 4, features * 8)
        self.layer5 = self.encoder(features * 8, features * 8)
        self.layer6 = self.encoder(features * 8, features * 8)
        self.layer7 = self.encoder(features * 8, features * 8)
        # self.latent = self.encoder(
        #     features * 8, features * 8, need_batch_norm=False)
        self.latent = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4,
                      stride=2, padding=1),
            nn.ReLU(),
        )

        self.layer8 = self.decoder(features * 8, features * 8, is_relu=True)
        self.layer9 = self.decoder(
            features * 8 * 2, features * 8, is_relu=True)
        self.layer10 = self.decoder(
            features * 8 * 2, features * 8, is_relu=True)
        self.layer11 = self.decoder(
            features * 8 * 2, features * 8, is_relu=True, need_dropout=False)
        self.layer12 = self.decoder(
            features * 8 * 2, features * 4, is_relu=True, need_dropout=False)
        self.layer13 = self.decoder(
            features * 4 * 2, features * 2, is_relu=True, need_dropout=False)
        self.layer14 = self.decoder(
            features * 2 * 2, features, is_relu=True, need_dropout=False)

        self.layer15 = nn.Sequential(
            nn.ConvTranspose2d(features*2, in_channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)

        latent = self.latent(layer7)
        layer8 = self.layer8(latent)
        layer9 = self.layer9(torch.cat([layer8, layer7], 1))
        layer10 = self.layer10(torch.cat([layer9, layer6], 1))
        layer11 = self.layer11(torch.cat([layer10, layer5], 1))
        layer12 = self.layer12(torch.cat([layer11, layer4], 1))
        layer13 = self.layer13(torch.cat([layer12, layer3], 1))
        layer14 = self.layer14(torch.cat([layer13, layer2], 1))

        return self.layer15(torch.cat([layer14, layer1], 1))

# global class for constants and hyperparameters


class config:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    TRAIN_DIR = "data/daynight/train"
    VAL_DIR = "data/daynight/val"
    LEARNING_RATE = 0.0002
    BATCH_SIZE = 16
    NUM_WORKERS = 2
    LAMBDA = 100
    NUM_EPOCHS = 50
    LOAD_MODEL = False
    SAVE_MODEL = True
    FLIP_TRAIN = False
    CHECKPOINT_DISC = "disc.pth.tar"
    CHECKPOINT_GEN = "gen.pth.tar"
    MODEL_DEFAULT = 'maps'
    MODEL_ANIME = 'anime'
    MODEL_DAYNIGHT = 'daynight'
    MODE = 'train'


class DataTransformation:
    resize = albumentations.Compose(
        [albumentations.Resize(width=256, height=256), ], additional_targets={"image0": "image"},
    )
    transform = albumentations.Compose(
        [
            albumentations.HorizontalFlip(p=0.5),
            albumentations.ColorJitter(p=0.2),
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[
                0.5, 0.5, 0.5], max_pixel_value=255.0,),
            ToTensorV2(),
        ]
    )
    tranform_mask = albumentations.Compose(
        [
            albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[
                0.5, 0.5, 0.5], max_pixel_value=255.0,),
            ToTensorV2(),
        ]
    )


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


'''
  This class extends the pytorch Dataset class
'''


class SplitData(Dataset):
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self) -> None:
        return len(self.list_files)

    def __getitem__(self, index) -> None:
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        # get the image shape
        image_dim = int(image.shape[1]/2)
        # print('image shape: ', image_dim)
        flip = config.FLIP_TRAIN
        if flip:
            target_image = image[:, :image_dim, :]
            input_image = image[:, image_dim:, :]
        else:
            input_image = image[:, :image_dim, :]
            target_image = image[:, image_dim:, :]

        augmentations = DataTransformation.resize(
            image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = DataTransformation.transform(image=input_image)["image"]
        target_image = DataTransformation.tranform_mask(image=target_image)[
            "image"]

        return input_image, target_image


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
) -> None:
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.LAMBDA
            # loss = torch.nn.HuberLoss()
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


# helper functions
def _getTrainDirectoryPath(modelname):
    return 'data/'+modelname+'/train' if modelname != None or modelname != '' else 'data/maps/train'


def _getValDirectoryPath(modelname):
    return 'data/'+modelname+'/val' if modelname != None or modelname != '' else 'data/maps/val'


def _getDiscCheckpointPath(modelname):
    return modelname+'_'+config.CHECKPOINT_DISC if modelname != None or modelname != '' else 'maps_'+config.CHECKPOINT_DISC


def _getGenCheckpointPath(modelname):
    return modelname+'_'+config.CHECKPOINT_GEN if modelname != None or modelname != '' else 'maps_'+config.CHECKPOINT_GEN


def main(args) -> None:
    # get data from the command line arguments
    config.LOAD_MODEL = True if args.mode == True else False
    config.FLIP_TRAIN = True if args.flip.lower() == 'true' else False
    config.NUM_EPOCHS = int(
        args.epochs) if args.epochs != None else config.NUM_EPOCHS
    config.MODE = args.mode if args.mode != None else config.MODE

    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(
        disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(
        gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE,
        )

    train_dataset = SplitData(root_dir=_getTrainDirectoryPath(args.modelname))
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = SplitData(root_dir=_getValDirectoryPath(args.modelname))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    val_itr = iter(val_loader)

    for epoch in range(1, config.NUM_EPOCHS+1):
        print('Epoch: {}/{}'.format(epoch, config.NUM_EPOCHS))

        if(config.MODE == 'train'):
            train_fn(
                disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
            )

        if config.SAVE_MODEL and epoch % 5 == 0:
            save_checkpoint(
                gen, opt_gen, filename=_getGenCheckpointPath(args.modelname))
            save_checkpoint(
                disc, opt_disc, filename=_getDiscCheckpointPath(args.modelname))

        x, y = next(val_itr)
        # get_test_samples(gen, x, y, epoch, folder="evaluation")
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        folder = "evaluation"
        gen.eval()
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5
            save_image(y_fake, folder + f"/y_gen_{epoch}.png")
            save_image(x * 0.5 + 0.5, folder + f"/input_{epoch}.png")
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
        gen.train()


if __name__ == "__main__":
    # setting up the argument parser to parse the command line arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--flip", default=False,
                           help="learn the left side of the image")
    argparser.add_argument(
        "--modelname", default=config.MODEL_DEFAULT, help="which model to load")
    argparser.add_argument("--mode", default='test',
                           help='start in train or test mode')
    argparser.add_argument("--epochs", default=50,
                           help="number of epochs to train")
    args = argparser.parse_args()
    print(args)

    # run the main function with all the passed command line arguments
    main(args)
