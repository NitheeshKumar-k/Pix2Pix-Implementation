#!/usr/bin/env python3
"""
Pix2Pix (maps) training with SpeechBrain-style recipe & logging.

- Model: U-Net generator + PatchGAN discriminator
- Loss: GAN (BCE with logits) + λ * L1
- Dataset: 'maps' aligned AB JPGs (each image is [A|B] horizontally)
- Logging: SpeechBrain FileTrainLogger + periodic sample image dumps
- Checkpointing: SpeechBrain Checkpointer (keeps top by valid L1)

Usage:
    python train.py receipt.yaml
    python train.py receipt.yaml --data_folder /path/to/maps

Directory layout (as in pix2pix repo):
    data_folder/
      train/*.jpg
      val/*.jpg
      (optional) test/*.jpg
Each JPG is concatenated horizontally: left=domain A, right=domain B.

Author: you & ChatGPT
"""
import os
import sys
import math
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random

import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import if_main_process
from speechbrain.utils.logger import get_logger
from torchvision.utils import save_image

logger = get_logger(__name__)


# ----------------------------- Dataset -----------------------------
class MapsAlignedDataset(Dataset):
    """Loads horizontally-concatenated AB images (maps dataset)."""

    def __init__(self, root, split, direction="AtoB", load_size=286, crop_size=256, random_flip=True):
        super().__init__()
        self.root = Path(root) / split
        self.paths = sorted([p for p in self.root.glob("*.jpg")])
        assert len(self.paths) > 0, f"No .jpg found in {self.root}"
        self.direction = direction
        self.load_size = load_size
        self.crop_size = crop_size
        self.random_flip = random_flip and (split == "train")

    def __len__(self):
        return len(self.paths)

    def _split_ab(self, img: Image.Image) -> Tuple[Image.Image, Image.Image]:
        w, h = img.size
        assert w % 2 == 0, f"Image width {w} not even for {self.paths[0].name}"
        w2 = w // 2
        A = img.crop((0, 0, w2, h))        # left
        B = img.crop((w2, 0, w, h))        # right
        return (A, B)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        A, B = self._split_ab(img)

        if self.direction == "AtoB":
            src, tgt = A, B
        else:
            src, tgt = B, A

        # Resize (load_size), RandomCrop (crop_size), RandomFlip
        if self.load_size != self.crop_size:
            src = src.resize((self.load_size, self.load_size), Image.BICUBIC)
            tgt = tgt.resize((self.load_size, self.load_size), Image.BICUBIC)

        # Random crop (train) else center crop
        if self.crop_size < self.load_size:
            if self.random_flip:  # using as proxy for "train split"
                x = random.randint(0, self.load_size - self.crop_size)
                y = random.randint(0, self.load_size - self.crop_size)
            else:
                x = (self.load_size - self.crop_size) // 2
                y = (self.load_size - self.crop_size) // 2
            box = (x, y, x + self.crop_size, y + self.crop_size)
            src = src.crop(box)
            tgt = tgt.crop(box)

        # Random horizontal flip (train only)
        if self.random_flip and random.random() < 0.5:
            src = src.transpose(Image.FLIP_LEFT_RIGHT)
            tgt = tgt.transpose(Image.FLIP_LEFT_RIGHT)

        # To tensor in [-1, 1]
        src = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(src.tobytes()))
                                .view(src.size[1], src.size[0], 3)
                                .numpy().astype('float32') / 255.0)).permute(2, 0, 1)
        tgt = torch.from_numpy((torch.ByteTensor(torch.ByteStorage.from_buffer(tgt.tobytes()))
                                .view(tgt.size[1], tgt.size[0], 3)
                                .numpy().astype('float32') / 255.0)).permute(2, 0, 1)

        src = src * 2.0 - 1.0
        tgt = tgt * 2.0 - 1.0

        return {"src": src, "tgt": tgt, "id": path.stem}


# ----------------------------- Models -----------------------------
def conv_block(in_c, out_c, norm=True, down=True):
    ks, st, pad = (4, 2, 1) if down else (4, 2, 1)
    layers = [nn.Conv2d(in_c, out_c, ks, st, pad, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_c))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """U-Net generator as in pix2pix (8 down/ups)."""
    def __init__(self, in_c=3, out_c=3, nf=64):
        super().__init__()
        # Encoder
        self.e1 = nn.Conv2d(in_c, nf, 4, 2, 1)             # no norm, LReLU later
        self.e2 = conv_block(nf, nf*2)                      # 128
        self.e3 = conv_block(nf*2, nf*4)                    # 256
        self.e4 = conv_block(nf*4, nf*8)                    # 512
        self.e5 = conv_block(nf*8, nf*8)                    # 512
        self.e6 = conv_block(nf*8, nf*8)                    # 512
        self.e7 = conv_block(nf*8, nf*8)                    # 512
        self.e8 = nn.Conv2d(nf*8, nf*8, 4, 2, 1)            # bottleneck no norm

        # Decoder (with dropout on first 3)
        def up(in_c, out_c, drop=False):
            layers = [nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                      nn.BatchNorm2d(out_c),
                      nn.ReLU(True)]
            if drop:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        self.d1 = up(nf*8, nf*8, drop=True)
        self.d2 = up(nf*16, nf*8, drop=True)
        self.d3 = up(nf*16, nf*8, drop=True)
        self.d4 = up(nf*16, nf*8)
        self.d5 = up(nf*16, nf*4)
        self.d6 = up(nf*8, nf*2)
        self.d7 = up(nf*4, nf)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(nf*2, out_c, 4, 2, 1),
            nn.Tanh()
        )

        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        e1 = self.lrelu(self.e1(x))
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)

        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        out = self.d8(d7)
        return out


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN (default)"""
    def __init__(self, in_c=3, nf=64, n_layers=3):
        super().__init__()
        layers = [nn.Conv2d(in_c * 2, nf, 4, 2, 1), nn.LeakyReLU(0.2, True)]
        ch = nf
        for i in range(1, n_layers):
            layers += [nn.Conv2d(ch, ch*2, 4, 2, 1, bias=False),
                       nn.BatchNorm2d(ch*2),
                       nn.LeakyReLU(0.2, True)]
            ch *= 2
        layers += [nn.Conv2d(ch, ch*2, 4, 1, 1, bias=False),
                   nn.BatchNorm2d(ch*2),
                   nn.LeakyReLU(0.2, True)]
        layers += [nn.Conv2d(ch*2, 1, 4, 1, 1)]  # no sigmoid (use BCEWithLogits)
        self.net = nn.Sequential(*layers)

    def forward(self, src, tgt):
        # input is concatenation of source and (real or fake) target
        x = torch.cat([src, tgt], dim=1)
        return self.net(x)


# ----------------------------- Brain -----------------------------
class Pix2PixBrain(sb.core.Brain):
    def compute_forward(self, batch, stage):
        batch = batch.to(self.device)
        src = batch.src
        tgt = batch.tgt

        fake = self.modules.G(src)
        # Discriminator outputs
        d_real = self.modules.D(src, tgt)
        d_fake = self.modules.D(src, fake.detach())
        return fake, d_real, d_fake, src, tgt

    def compute_objectives(self, predictions, batch, stage):
        fake, d_real, d_fake, src, tgt = predictions

        # GAN losses
        valid = torch.ones_like(d_real)
        fake_lbl = torch.zeros_like(d_fake)

        gan_loss_G = self.hparams.bce_with_logits(self.modules.D(src, fake), valid)
        l1_loss = F.l1_loss(fake, tgt) * self.hparams.lambda_L1
        g_loss = gan_loss_G + l1_loss

        # D loss
        d_loss_real = self.hparams.bce_with_logits(d_real, valid)
        d_loss_fake = self.hparams.bce_with_logits(d_fake, fake_lbl)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Track
        self.gan_loss = gan_loss_G.detach()
        self.l1_loss = (l1_loss.detach() / self.hparams.lambda_L1)  # raw L1
        self.d_loss = d_loss.detach()

        # Optimizer switching handled in fit_batch()
        return g_loss, d_loss, fake, src, tgt

    def fit_batch(self, batch):
        self.optimizer_G.zero_grad(set_to_none=True)
        self.optimizer_D.zero_grad(set_to_none=True)

        fake, d_real, d_fake, src, tgt = self.compute_forward(batch, sb.Stage.TRAIN)
        g_loss, d_loss, _, _, _ = self.compute_objectives((fake, d_real, d_fake, src, tgt), batch, sb.Stage.TRAIN)

        # Update G
        self.scaler.scale(g_loss).backward(retain_graph=True)
        self.scaler.step(self.optimizer_G)

        # Update D
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.optimizer_D)

        self.scaler.update()
        return (g_loss + d_loss).detach()

    def evaluate_batch(self, batch, stage):
        with torch.no_grad():
            fake, d_real, d_fake, src, tgt = self.compute_forward(batch, stage)
            g_loss, d_loss, _, _, _ = self.compute_objectives((fake, d_real, d_fake, src, tgt), batch, stage)
            return (g_loss + d_loss).detach()

    def on_stage_end(self, stage, stage_loss, epoch):
        stats = {
            "loss": float(stage_loss),
            "G_GAN": float(self.gan_loss.mean().cpu().item()) if hasattr(self, "gan_loss") else None,
            "G_L1": float(self.l1_loss.mean().cpu().item()) if hasattr(self, "l1_loss") else None,
            "D": float(self.d_loss.mean().cpu().item()) if hasattr(self, "d_loss") else None,
        }
        if stage == sb.Stage.TRAIN:
            self.train_stats = stats
        elif stage == sb.Stage.VALID:
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch},
                train_stats=self.train_stats,
                valid_stats=stats,
            )
            # keep best by lowest valid L1 (proxy)
            key = -stats["G_L1"] if stats["G_L1"] is not None else -stats["loss"]
            self.checkpointer.save_and_keep_only(meta={"neg_L1": key, "epoch": epoch}, min_keys=["neg_L1"])
            # dump samples
            if if_main_process():
                self._save_samples(epoch, split="valid")
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )
            if if_main_process():
                # If val is reused as test, filename will reflect that
                split_name = "val_as_test" if self.hparams.get("use_val_as_test", False) else "test"
                self._save_samples(epoch, split=split_name)

    def _save_samples(self, epoch, split="valid"):
        # Save a tiny grid from last seen batch (stored in buffers)
        out_dir = Path(self.hparams.samples_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        # For simplicity, do a tiny re-run on a small subset from the split
        dl = self.sample_loaders["test" if split in ("test", "val_as_test") else "valid"]
        try:
            batch = next(iter(dl))
        except StopIteration:
            return
        batch = batch.to(self.device)
        with torch.no_grad():
            fake = self.modules.G(batch.src)
        # Denormalize from [-1,1] to [0,1]
        def denorm(x): return (x + 1.0) * 0.5
        grid = torch.cat([denorm(batch.src[:4]), denorm(fake[:4]), denorm(batch.tgt[:4])], dim=0)
        save_image(grid, out_dir / f"samples_{split}_epoch{epoch:04d}.png", nrow=4)

    @torch.no_grad()
    def on_fit_start(self):
        # build sample loaders for saving visuals
        self.sample_loaders = {}
        for split in ["valid", "test"]:
            if split in self.hparams.sample_loader_builders:
                self.sample_loaders[split] = self.hparams.sample_loader_builders[split]()


# ----------------------------- DataIO -----------------------------
def build_dataloaders(hparams):
    """Build dataloaders. If no 'test' folder exists, use 'val' as test set too."""
    def make(split, shuffle):
        ds = MapsAlignedDataset(
            root=hparams["data_folder"],
            split=split,
            direction=hparams["direction"],
            load_size=hparams["load_size"],
            crop_size=hparams["crop_size"],
            random_flip=(split == "train"),
        )
        return DataLoader(
            ds,
            batch_size=hparams["batch_size"] if split == "train" else 1,
            shuffle=shuffle,
            num_workers=hparams["num_workers"],
            pin_memory=True,
            drop_last=(split == "train"),
        )

    train_loader = make("train", shuffle=True)
    valid_loader = make("val", shuffle=False)

    test_path = os.path.join(hparams["data_folder"], "test")
    use_val_as_test = not os.path.isdir(test_path)
    if use_val_as_test:
        logger.info("No 'test' folder found — reusing 'val' split as the test set.")
        test_loader = valid_loader
    else:
        test_loader = make("test", shuffle=False)

    # tiny builders for sample saving inside Brain
    def make_builder(split):
        return lambda: make(split, shuffle=False)

    sample_loader_builders = {
        "valid": make_builder("val"),
        "test": (make_builder("val") if use_val_as_test else make_builder("test")),
    }

    # store flag in hparams for consistent naming/logging later
    hparams["use_val_as_test"] = use_val_as_test
    return train_loader, valid_loader, test_loader, sample_loader_builders


# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Seed & device
    torch.backends.cudnn.benchmark = True

    # Data
    train_loader, valid_loader, test_loader, sample_loader_builders = build_dataloaders(hparams)
    hparams["sample_loader_builders"] = sample_loader_builders

    # Modules
    G = UNetGenerator(in_c=3, out_c=3, nf=hparams["ngf"])
    D = PatchDiscriminator(in_c=3, nf=hparams["ndf"], n_layers=hparams["n_layers_D"])

    modules = {"G": G, "D": D}
    model_list = nn.ModuleList([G, D])

    # Opts
    optimizer_G = torch.optim.Adam(G.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], 0.999))
    optimizer_D = torch.optim.Adam(D.parameters(), lr=hparams["lr"], betas=(hparams["beta1"], 0.999))

    # Checkpointer
    checkpointer = sb.utils.checkpoints.Checkpointer(
        checkpoints_dir=hparams["save_folder"],
        recoverables={
            "model": model_list,
            "opt_G": optimizer_G,
            "opt_D": optimizer_D,
            "counter": hparams["epoch_counter"],
        },
    )

    # Brain
    brain = Pix2PixBrain(
        modules=modules,
        opt_class=None,  # we handle two opts manually
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=checkpointer,
    )
    # attach optimizers & scaler
    brain.optimizer_G = optimizer_G
    brain.optimizer_D = optimizer_D
    brain.scaler = torch.cuda.amp.GradScaler(enabled=(hparams.get("precision", "fp32") != "fp32"))

    # Train / Valid
    brain.fit(
        epoch_counter=hparams["epoch_counter"],
        train_set=train_loader,
        valid_set=valid_loader,
        train_loader_kwargs={},  # already DataLoader
        valid_loader_kwargs={},
    )

    # Test
    # If val is reused as test, the sample saver will name outputs "val_as_test"
    brain.evaluate(
        test_set=test_loader,
        max_key=None,
        test_loader_kwargs={},
    )
