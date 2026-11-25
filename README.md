# Part-Aware Few-Shot Fine-Grained Recognition
Research question: does spatial activation-weighted pooling improve frozen global prototypes?

## Setup
Python 3.10–3.12 recommended. Install `pip install -r requirements.txt`. Download CUB-200-2011 from https://www.vision.caltech.edu/datasets/cub_200_2011/ and extract so `data/CUB_200_2011/images` exists. The first run downloads torchvision ImageNet ResNet-18 weights. On Windows use Git Bash with the same Python environment. Run `bash run_all.sh` (or set CUB_ROOT to the extracted directory). Demo: `streamlit run app.py`. Cache weights before offline use.

## Protocol
Custom fixed 100/50/50 class split (seed 18); only 50 test classes are encoded. Frozen ImageNet backbone, no CUB training or validation tuning. Possible ImageNet semantic overlap is a limitation. Five classes, 1 or 5 supports, 15 disjoint queries per class, 200 episodes per seed, seeds 42/123/2026. Identical episodes across methods. Local pooling uses spatial activation magnitude, not verified anatomical parts. Combined score is equal-weight cosine fusion, fixed before testing. Approximate episode-level confidence intervals are descriptive, not independent dataset-level uncertainty.

## Results and hardware
Pending execution. Generated CSVs, figures, logs and hardware/runtime metadata appear in results/. Commit actual outputs only. Do not report unrun numbers. Compare global cosine against combined and local-only ablation. Inspect actual failures using failure_pairs.csv and failures.csv.

## Credits
CUB: Wah et al., The Caltech-UCSD Birds-200-2011 Dataset (2011), dataset link above. ResNet: He et al., Deep Residual Learning for Image Recognition (CVPR 2016). Torchvision model and pretrained checkpoint: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html . Prototype concept: Snell et al., Prototypical Networks for Few-shot Learning (NeurIPS 2017), https://arxiv.org/abs/1703.05175 . No external baseline repository copied. AI-generated starter implementation; see AI_DECLARATION.md.

## Proposal changes
Frozen backbone rather than training on base classes; simple spatial saliency pooling rather than semantic part detection; cosine vs Euclidean baseline rather than a separately trained stronger backbone. Secondary datasets deferred. Gates and hypothesis outcome must be updated after actual execution.
