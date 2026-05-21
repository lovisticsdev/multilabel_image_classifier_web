from __future__ import annotations

from PIL import Image, ImageOps

from multilabel_classifier.constants import IMAGENET_MEAN, IMAGENET_STD


def preprocess_image(image: Image.Image, image_size: int):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    image = ImageOps.exif_transpose(image).convert("RGB")
    return transform(image).unsqueeze(0)
