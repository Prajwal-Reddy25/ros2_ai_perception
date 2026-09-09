# Training scope

Training is intentionally not part of the core workflow. This release uses
official torchvision COCO-pretrained weights and includes no dataset or
fine-tuned checkpoint. The inference, ROS integration, benchmarking, and small
dataset evaluation interfaces are complete without training.

Fine-tuning should be added only alongside a verified dataset license, a fixed
split manifest, deterministic configuration, checkpoint provenance, and
held-out evaluation. No accuracy claim in this repository is based on custom
training.
