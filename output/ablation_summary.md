# Layer Ablation Study Results (Layers 6, 7, 8, 9)

This table summarizes the segment-level classification metrics and speaker-level majority vote results across all four WavLM layers.

| Layer | Config | Model | Segment Accuracy | Segment F1 | Segment AUC | Speaker Vote (MDD/HC Correct) |
| :---: | :--- | :--- | :---: | :---: | :---: | :--- |
| 6 | EN -> EN | Baseline | 0.7379 | 0.6628 | 0.8043 | - |
| 6 | EN -> EN | CLeaD | 0.7411 | 0.6148 | 0.7704 | - |
| 6 | EN -> ZH | Baseline | 0.4958 | 0.2545 | 0.4807 | 0/5 MDD, 5/5 HC |
| 6 | EN -> ZH | CLeaD | 0.5013 | 0.2833 | 0.5255 | 0/5 MDD, 5/5 HC |
| 6 | ZH -> EN | Baseline | 0.5432 | 0.4217 | 0.5357 | - |
| 6 | ZH -> EN | CLeaD | 0.5535 | 0.3490 | 0.5539 | - |
| 6 | ZH -> ZH | Baseline | 0.5363 | 0.4411 | 0.5368 | 1/5 MDD, 5/5 HC |
| 6 | ZH -> ZH | CLeaD | 0.5414 | 0.4488 | 0.5721 | 1/5 MDD, 5/5 HC |
| 6 | MIX -> EN | Baseline | 0.6695 | 0.5453 | 0.7023 | - |
| 6 | MIX -> EN | CLeaD | 0.6934 | 0.5035 | 0.7088 | - |
| 6 | MIX -> ZH | Baseline | 0.5038 | 0.4272 | 0.4809 | 2/5 MDD, 4/5 HC |
| 6 | MIX -> ZH | CLeaD | 0.5685 | 0.5386 | 0.5855 | 4/5 MDD, 4/5 HC |
| 7 | EN -> EN | Baseline | 0.7190 | 0.6439 | 0.7811 | - |
| 7 | EN -> EN | CLeaD | 0.7323 | 0.6180 | 0.7799 | - |
| 7 | EN -> ZH | Baseline | 0.4678 | 0.3323 | 0.4725 | 0/5 MDD, 5/5 HC |
| 7 | EN -> ZH | CLeaD | 0.4900 | 0.3886 | 0.5013 | 0/5 MDD, 5/5 HC |
| 7 | ZH -> EN | Baseline | 0.5578 | 0.3687 | 0.5609 | - |
| 7 | ZH -> EN | CLeaD | 0.5377 | 0.3371 | 0.5070 | - |
| 7 | ZH -> ZH | Baseline | 0.5221 | 0.4521 | 0.5193 | 2/5 MDD, 5/5 HC |
| 7 | ZH -> ZH | CLeaD | 0.5322 | 0.3946 | 0.5182 | 2/5 MDD, 5/5 HC |
| 7 | MIX -> EN | Baseline | 0.6790 | 0.5646 | 0.7043 | - |
| 7 | MIX -> EN | CLeaD | 0.6687 | 0.4281 | 0.6443 | - |
| 7 | MIX -> ZH | Baseline | 0.4720 | 0.4352 | 0.4570 | 3/5 MDD, 4/5 HC |
| 7 | MIX -> ZH | CLeaD | 0.4950 | 0.4537 | 0.4928 | 2/5 MDD, 4/5 HC |
| 8 | EN -> EN | Baseline | 0.6823 | 0.6083 | 0.7454 | - |
| 8 | EN -> EN | CLeaD | 0.6551 | 0.4913 | 0.6669 | - |
| 8 | EN -> ZH | Baseline | 0.4900 | 0.3994 | 0.4919 | 0/5 MDD, 5/5 HC |
| 8 | EN -> ZH | CLeaD | 0.5104 | 0.3453 | 0.5467 | 0/5 MDD, 5/5 HC |
| 8 | ZH -> EN | Baseline | 0.5467 | 0.3591 | 0.5362 | - |
| 8 | ZH -> EN | CLeaD | 0.5442 | 0.3804 | 0.5221 | - |
| 8 | ZH -> ZH | Baseline | 0.4987 | 0.4112 | 0.5006 | 2/5 MDD, 5/5 HC |
| 8 | ZH -> ZH | CLeaD | 0.5205 | 0.4202 | 0.5322 | 2/5 MDD, 5/5 HC |
| 8 | MIX -> EN | Baseline | 0.6692 | 0.5643 | 0.7004 | - |
| 8 | MIX -> EN | CLeaD | 0.5950 | 0.4333 | 0.5697 | - |
| 8 | MIX -> ZH | Baseline | 0.4616 | 0.4339 | 0.4513 | 2/5 MDD, 3/5 HC |
| 8 | MIX -> ZH | CLeaD | 0.4937 | 0.3841 | 0.5086 | 2/5 MDD, 5/5 HC |
| 9 | EN -> EN | Baseline | 0.6834 | 0.6165 | 0.7545 | - |
| 9 | EN -> EN | CLeaD | 0.6712 | 0.5184 | 0.6953 | - |
| 9 | EN -> ZH | Baseline | 0.4724 | 0.2294 | 0.4637 | 0/5 MDD, 5/5 HC |
| 9 | EN -> ZH | CLeaD | 0.5113 | 0.2935 | 0.5562 | 0/5 MDD, 5/5 HC |
| 9 | ZH -> EN | Baseline | 0.5442 | 0.2645 | 0.5481 | - |
| 9 | ZH -> EN | CLeaD | 0.5127 | 0.2803 | 0.4606 | - |
| 9 | ZH -> ZH | Baseline | 0.5201 | 0.4573 | 0.5082 | 2/5 MDD, 5/5 HC |
| 9 | ZH -> ZH | CLeaD | 0.5226 | 0.4356 | 0.5235 | 2/5 MDD, 5/5 HC |
| 9 | MIX -> EN | Baseline | 0.6755 | 0.5668 | 0.7054 | - |
| 9 | MIX -> EN | CLeaD | 0.6093 | 0.4173 | 0.5784 | - |
| 9 | MIX -> ZH | Baseline | 0.4620 | 0.4415 | 0.4476 | 2/5 MDD, 3/5 HC |
| 9 | MIX -> ZH | CLeaD | 0.5263 | 0.4933 | 0.5395 | 3/5 MDD, 4/5 HC |
