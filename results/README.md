# Results

Outputs of the experiments reported in the paper. Every folder name matches the output folder of the notebook that produced it.

| Folder | Contents |
|---|---|
| `results_gopro/` | Full model on GoPro (100 epochs): ten test samples (input, output, ground truth), training curves, training log |
| `results_hide/` | Full model on HIDE (35 epochs): ten test samples, training curves, validation metrics |
| `results_realblurj/` | Full model on RealBlur-J (70 epochs): ten test samples, training curves, validation metrics |
| `ablation_figures/` | Sample output and training curves of each ablation run |
| `ablation_json/` | Test evaluations and training histories of the ablation runs |
| `split_proofs/` | SHA256 hashes of the train, validation and test file lists of every run, checked by `scripts/verify_splits.py` |

Notes:

- The ten test images in each `results_*` folder were picked at random by the notebook, and the PSNR and SSIM printed on each belong to that single image, not to the dataset average.
- `results_gopro/sample_3_comparison.png` was recaptured at a lower resolution because the original render of that file was corrupted.
- The notebooks that produced these runs are in `train/` (full models) and `ablation/` (ablations).
