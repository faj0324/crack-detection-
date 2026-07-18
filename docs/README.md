# docs folder

Put the two images from the training run in here so they show up in the Results
section of the main [README](../README.md):

| File                  | What it is                                          |
| --------------------- | --------------------------------------------------- |
| `results.png`         | The training graphs (loss and mAP over the epochs). |
| `val_batch0_pred.jpg` | The model's predictions on a batch of val images.   |

Both come out of the training run under `runs/detect/train/`. Copy them here
with these exact names:

```bash
cp runs/detect/train/results.png            docs/results.png
cp runs/detect/train/val_batch0_pred.jpg    docs/val_batch0_pred.jpg
```

The trained weights (`best.pt`) don't go here, they go in the
[`models/`](../models/) folder.
