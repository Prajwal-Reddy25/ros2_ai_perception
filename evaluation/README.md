# Evaluation interface

The evaluator runs the selected pretrained model and reports AP at IoU 0.50,
plus micro precision and recall. No dataset is bundled or automatically
assumed. Metrics are valid only for the labeled manifest supplied by the user.

Create a JSON manifest beside its images:

```json
{
  "images": [
    {
      "id": "frame-001",
      "file": "images/frame-001.jpg",
      "annotations": [
        {"label": "person", "bbox_xyxy": [45, 30, 220, 410]}
      ]
    }
  ]
}
```

Labels must use the torchvision COCO category names and boxes use pixel-space
`[x1, y1, x2, y2]`. Run:

```bash
perception_evaluate evaluation/manifest.json \
  --output results/evaluation.json --device cpu
```

The implementation uses score-ordered greedy one-to-one matching at IoU 0.50
and interpolated precision/recall integration. This is intentionally compact
for small licensed datasets; use the official COCO API for publication-grade
COCO metrics over COCO-format datasets. The compact evaluator scores only
classes present in the ground-truth manifest. Predictions belonging only to
classes absent from that manifest are outside its scoring scope.

