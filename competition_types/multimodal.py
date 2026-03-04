"""
Multimodal competition specialist.
Cross-modal fusion strategies for image+text, tabular+image competitions.
"""

MULTIMODAL_SYSTEM_CONTEXT = """
You are in multimodal specialist mode.

Fusion architectures:
1. Late fusion (most robust, recommended default):
   - Train separate specialists for each modality
   - Combine final predictions via weighted blend or stacking
   - Pros: modular, easy to debug, each model can be optimized independently
   - Cons: doesn't capture cross-modal interactions

2. Early fusion:
   - Concatenate features from different modalities before final layers
   - Extract image features (EfficientNet/ViT embeddings) + text features (DeBERTa embeddings)
   - Concatenate → MLP head
   - Pros: captures interactions, single model
   - Cons: harder to train, more hyperparameters

3. Cross-attention fusion (best, hardest):
   - Use cross-attention between image patch tokens and text tokens
   - CLIP-style or FLAVA architecture
   - Only worth it if you have sufficient training data (>10k examples)

Strategy by competition type:
- Product classification (image + title text): Late fusion LGBM + fine-tuned CLIP
- Medical report classification (image + report text): Cross-attention, domain-pretrained models
- Real estate valuation (tabular + images): LightGBM on tabular + CNN on images, late fusion
- Social media engagement (text + image): CLIP embeddings + LGBM

CLIP embedding strategy (often wins multimodal tabular competitions):
  from transformers import CLIPModel, CLIPProcessor
  model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  # Extract 768-dim image embeddings and 768-dim text embeddings
  # Concatenate (1536-dim) → feed into LightGBM or neural head
  # CLIP is already cross-modally aligned — embeddings are directly comparable

Common pitfalls:
- Training end-to-end from scratch with small multimodal datasets (<5k)
- Not normalizing image and text embedding magnitudes before fusion
- Forgetting that different modalities may have different missing rates
- Using the same augmentations for images as standalone CV (may corrupt text/context)
"""

FUSION_RECIPES = {
    "image_text_classification": {
        "image_backbone": "openai/clip-vit-large-patch14",
        "text_backbone": "microsoft/deberta-v3-base",
        "fusion": "late",
        "meta_model": "lightgbm",
    },
    "tabular_image": {
        "image_backbone": "efficientnet_b4",
        "tabular_model": "lightgbm",
        "fusion": "late",
        "blend_ratio": 0.6,  # 60% tabular, 40% image
    },
    "vlm_zero_shot": {
        "model": "llava-v1.6-mistral-7b",
        "strategy": "zero_shot_then_finetune",
        "use_when": "very_small_dataset_under_500_examples",
    },
}
