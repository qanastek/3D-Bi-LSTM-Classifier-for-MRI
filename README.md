# Identification of COVID-19 cases using a 3D Bi-LSTM Classifier for Chest CT scans

All the Chest CT scans are taken from the [MosMedData](https://www.medrxiv.org/content/10.1101/2020.05.20.20100362v1) dataset.

|            | CT-0 | CT-23 |
|------------|------|-------|
| Raw (512x512) | <img src="gifs/ct0_normal_raw.gif" width="256" height="256" /> | <img src="gifs/ct23_abnormal_raw.gif" width="256" height="256" /> |
| Normalized (256x256 + Depthness 64) | <img src="gifs/ct0_normal_normalized.gif" width="256" height="256" /> | <img src="gifs/ct23_abnormal_normalized.gif" width="256" height="256" /> |

## Results

```plain
              precision    recall  f1-score   support

      normal       0.75      0.90      0.82        10
    abnormal       0.88      0.70      0.78        10

    accuracy                           0.80        20
   macro avg       0.81      0.80      0.80        20
weighted avg       0.81      0.80      0.80        20
```

## References

MosMedData: Chest CT Scans with COVID-19 Related Findings Dataset

```bibtex
@misc{morozov2020mosmeddata,
title={MosMedData: Chest CT Scans With COVID-19 Related Findings Dataset}, 
author={S. P. Morozov and A. E. Andreychenko and N. A. Pavlov and A. V. Vladzymyrskyy and N. V. Ledikhova and V. A. Gombolevskiy and I. A. Blokhin and P. B. Gelezhe and A. V. Gonchar and V. Yu. Chernina},
year={2020},
eprint={2005.06465},
archivePrefix={arXiv},
primaryClass={cs.CY}
}
```
