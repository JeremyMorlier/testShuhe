import torch
from functools import partial
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from argparse import ArgumentParser
import utils
from torchvision.transforms.functional import InterpolationMode
import sys
import time
from torch.utils.data.dataloader import default_collate
import temp.backbones2 as backbones

# sys.path.insert(0, "/home/tsuyuki/research/brain-train/KD/dinov2")
# from dinov2.models.vision_transformer import vit_base, vit_large, vit_small
import os
import presets as presets
from PIL import Image
import warnings

warnings.filterwarnings("ignore")

parser = ArgumentParser("Main")
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--weight-decay", type=float, default=1e-3)
parser.add_argument("--model", type=str, default="vit-s")
parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--batch-size", type=int, default=376)
parser.add_argument("--save", action="store_true", help="save the student model")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument(
    "--data_path",
    default="/nasbrain/j20morli/eval_clip/imagenet/",
    type=str,
    help="dataset path",
)
parser.add_argument(
    "--val_crop_size",
    default=84,
    type=int,
    help="the central crop size used for validation (default: 224)",
)
parser.add_argument(
    "--train_crop_size",
    default=84,
    type=int,
    help="the random crop size used for training (default: 224)",
)
parser.add_argument(
    "--val_resize_size",
    default=84,
    type=int,
    help="the resize size used for validation (default: 256)",
)
parser.add_argument(
    "--student_input_size",
    default=84,
    type=int,
    help="input size for the student model",
)
parser.add_argument(
    "--interpolation",
    default="bilinear",
    type=str,
    help="the interpolation method (default: bilinear)",
)
parser.add_argument(
    "--cache_dataset",
    dest="cache_dataset",
    help="Cache the datasets for quicker initialization. It also serializes the transforms",
    action="store_true",
)
parser.add_argument(
    "--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive"
)
parser.add_argument("--use_v2", action="store_true", help="Use V2 transforms")
parser.add_argument(
    "--weights", default=None, type=str, help="the weights enum name to load"
)
parser.add_argument(
    "--test_only", dest="test_only", help="Only test the model", action="store_true"
)
parser.add_argument(
    "-j",
    "--workers",
    default=16,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 16)",
)

# addition
parser.add_argument(
    "--backbone", type=str, default="mobilevit_xxs", help="backbone architecture"
)
parser.add_argument(
    "--save-features-prefix",
    type=str,
    default="",
    help="save features of validation and test sets to hard drive, use this parameter as prefix to file names",
)
parser.add_argument(
    "--save-backbone",
    type=str,
    default="",
    help="save backbone to hard drive at the specified location",
)
parser.add_argument("--device", type=str, default="cuda:0", help="device to use")

args = parser.parse_args()

models = {}
models["vit-s"] = {
    "path": "/Brain",
    "embed_dim": 384,
    "n": [11],
}
models["vit-b"] = {
    "path": "/Brain/private/r17bensa/models/dinov2_vitb14_pretrain.pth",
    "embed_dim": 768,
    "n": [11],
}
models["vit-l"] = {
    "path": "/Brain/private/r17bensa/models/dinov2_vitl14_pretrain.pth",
    "embed_dim": 1024,
    "n": [23],
}


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
        args.train_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading training data")
    st = time.time()
    cache_path = utils._get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # We need a default value for the variables below because args may come
        # from train_quantization.py which doesn't define them.
        auto_augment_policy = getattr(args, "auto_augment", None)
        random_erase_prob = getattr(args, "random_erase", 0.0)
        ra_magnitude = getattr(args, "ra_magnitude", None)
        augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            presets.ClassificationPresetTrain(
                crop_size=train_crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=args.backend,
                use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = utils._get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        # TODO: this could probably be weights_only=True
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        if args.weights and args.test_only:
            weights = torchvision.models.get_weight(args.weights)
            preprocessing = weights.transforms(antialias=True)
            if args.backend == "tensor":
                preprocessing = torchvision.transforms.Compose(
                    [torchvision.transforms.PILToTensor(), preprocessing]
                )

        else:
            preprocessing = presets.ClassificationPresetEval(
                crop_size=val_crop_size,
                resize_size=val_resize_size,
                interpolation=interpolation,
                backend=args.backend,
                use_v2=args.use_v2,
            )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")

    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler


train_dir = os.path.join(args.data_path, "train")
val_dir = os.path.join(args.data_path, "val")
dataset, dataset_test, train_sampler, test_sampler = load_data(train_dir, val_dir, args)
collate_fn = default_collate

train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    sampler=train_sampler,
    num_workers=args.workers,
    pin_memory=True,
    collate_fn=collate_fn,
)
val_loader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=args.batch_size,
    sampler=test_sampler,
    num_workers=args.workers,
    pin_memory=True,
)


# Creating the teacher model
def create_backbone_dinov2(model):
    dino_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    # dino_backbone.load_state_dict(torch.load(models[model]["path"]), strict=False)
    n = models[model]["n"]
    dino_backbone.forward = partial(
        dino_backbone.get_intermediate_layers,
        n=n,
        reshape=True,
        return_class_token=True,
    )
    return dino_backbone


torch.manual_seed(seed=args.seed)
print(args)
# Loading and fixing the teacher model
teacher_model = create_backbone_dinov2(args.model)
teacher_model = teacher_model.to(args.device)

for param in teacher_model.parameters():
    param.requires_grad = False

student_model, outputDim = backbones.prepareBackbone(
    name="resnet12", feature_maps=32, pretrained=False
)
student_model = student_model.to(args.device)

adaptation_layer = nn.Linear(outputDim, models[args.model]["embed_dim"], bias=False)
adaptation_layer = adaptation_layer.to(args.device)

transform = transforms.Resize((args.student_input_size, args.student_input_size))
# Loss function
loss_fn = nn.MSELoss()
# Optimaizer
optimizer = torch.optim.AdamW(
    list(student_model.parameters()) + list(adaptation_layer.parameters()),
    lr=float(args.lr),
    weight_decay=args.weight_decay,
)
# Scheduler
scheduler = CosineAnnealingLR(
    optimizer, T_max=args.num_epochs, eta_min=0, last_epoch=-1
)

# Initialize learning history list
loss_hist_train = [0.0] * args.num_epochs
accuracy_hist_train = [0.0] * args.num_epochs
loss_hist_valid = [0.0] * args.num_epochs
accuracy_hist_valid = [0.0] * args.num_epochs

# Learning loop (Supervised distillation learning)
print("\n")
for epoch in range(args.num_epochs):
    epoch_start_time = time.time()
    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    teacher_model.eval()
    student_model.train()
    loss_accumulated_train = 0.0
    total_samples_train = 0

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader, start=1):
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        optimizer.zero_grad()

        with torch.no_grad():
            output_teacher = teacher_model(x_batch)[0][1]

        x_batch = transform(x_batch)
        output_student = student_model(x_batch)
        output_student = adaptation_layer(output_student)

        loss = loss_fn(output_student, output_teacher)
        loss.backward()
        optimizer.step()
        loss_accumulated_train += loss.item() * y_batch.size(0)
        total_samples_train += y_batch.size(0)

        print(
            f"Epoch {epoch + 1}/{args.num_epochs} - Batch {batch_idx}/{len(train_loader)}: Loss = {loss.item():.4f}"
        )

    scheduler.step()

    loss_accumulated_val = 0.0
    total_samples_val = 0

    for x_batch, y_batch in val_loader:
        x_batch = x_batch.to(args.device)
        y_batch = y_batch.to(args.device)

        with torch.no_grad():
            output_teacher = teacher_model(x_batch)[0][1]
            x_batch = transform(x_batch)
            output_student = student_model(x_batch)
            output_student = adaptation_layer(output_student)
        loss = loss_fn(output_student, output_teacher)
        loss_accumulated_val += loss.item() * y_batch.size(0)
        total_samples_val += y_batch.size(0)

    epoch_duration = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds.")

    torch.save(
        student_model.state_dict(),
        f"/Brain/private/j20morli/test/run2_{epoch + 1}.pth",
    )
