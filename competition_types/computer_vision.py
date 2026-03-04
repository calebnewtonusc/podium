"""
Computer vision competition specialist.
Architecture selection, augmentation strategy, TTA, pseudo-labeling for image competitions.
"""

CV_SYSTEM_CONTEXT = """
You are in computer vision specialist mode.

Architecture selection heuristics:
- Image classification (<224px): EfficientNetB4/B5 or ConvNeXt-Small
- Image classification (>224px, fine-grained): ViT-B/16 or EVA-02
- Medical imaging: EfficientNet + custom preprocessing (CLAHE, normalization by modality)
- Object detection: YOLOv8 / DINO / RT-DETR
- Segmentation: SegFormer / SAM with fine-tuning

Key techniques for top CV scores:
1. Progressive resizing: start small (128px), upscale during training
2. TTA (Test-Time Augmentation): horizontal flip + 4-crop minimum, 5-10 augmentations typical
3. Mixup/CutMix: especially effective for fine-grained classification
4. Cosine annealing LR with warm restarts
5. Label smoothing (0.1) for classification
6. Pseudo-labeling: high-confidence test predictions added to training (threshold 0.95+)
7. Multi-scale training for detection/segmentation

Augmentation stack (strong):
  - RandomHorizontalFlip
  - RandomVerticalFlip (if orientation-invariant)
  - RandomRotation(15)
  - ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
  - RandomResizedCrop
  - Normalize(imagenet_mean, imagenet_std)

CV strategy for images:
- StratifiedKFold on labels (not random split)
- 5 folds standard; 10 folds if dataset < 5k images
- Never shuffle medical imaging by patient — group by patient_id

Common pitfalls:
- Not normalizing with ImageNet stats when using pretrained weights
- Using augmentations that change the label (e.g., rotation in digit recognition)
- Batch size too large → poor generalization; prefer smaller batches with gradient accumulation
- Not freezing backbone early: fine-tune head first (2-3 epochs), then unfreeze all
"""

ARCHITECTURE_RECIPES = {
    "classification_small": {
        "model": "efficientnet_b4",
        "input_size": 224,
        "batch_size": 32,
        "lr": 1e-4,
        "epochs": 20,
        "augmentation": "medium",
    },
    "classification_large": {
        "model": "vit_base_patch16_224",
        "input_size": 384,
        "batch_size": 16,
        "lr": 5e-5,
        "epochs": 15,
        "augmentation": "strong",
    },
    "medical_imaging": {
        "model": "efficientnet_b5",
        "input_size": 512,
        "batch_size": 8,
        "lr": 1e-4,
        "epochs": 25,
        "augmentation": "medical",
        "preprocessing": "clahe",
    },
    "fine_grained": {
        "model": "convnext_base",
        "input_size": 384,
        "batch_size": 16,
        "lr": 5e-5,
        "epochs": 20,
        "augmentation": "strong",
        "use_tta": True,
    },
}
