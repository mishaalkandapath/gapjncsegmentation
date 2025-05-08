# Automatic Gap Junction Segmentation in the C.elegans
We present a supervised training pipeline for automatic gap junction annotation in the _C_. _elegans_. The pipeline consists of training a UNET-based model on certain parts of the worm, and then deploying a specialized inference-technique to combine predictions across cross-sections.
<p align="center">
<img width="60%" src="https://github.com/mishaalkandapath/gapjncsegmentation/blob/main/gapjncimages/workflow.png"><br>
  Overall workflow
</p>
<p align="center">
<img width="60%" src="https://github.com/mishaalkandapath/gapjncsegmentation/blob/main/gapjncimages/normalunet.png"><br>
  Model architecture
</p>

## Reproducing Results
Coming soon

### Annotation Results
<div align="center">
  <img src="https://github.com/mishaalkandapath/gapjncsegmentation/blob/main/gapjncimages/wormem.png" alt="Image 1" width="30%" style="display: inline-block; margin: 0 10px"/>
  <img src=https://github.com/mishaalkandapath/gapjncsegmentation/blob/main/gapjncimages/emannotatedmembrane.png alt="Image 2" width="30%" style="display: inline-block; margin: 0 10px"/>
  <img src=https://github.com/mishaalkandapath/gapjncsegmentation/blob/main/gapjncimages/emannotatedextend.png alt="Image 3" width="30%" style="display: inline-block; margin: 0 10px"/>
<br> Original EM Image - Predictions from UNET-based backbone - Predictions from UNET-based backbone + extension inference pipeline
</div>

### Overall Results
<div align="center">

| Model Name | Recall | Entity Recall<br>(50%, 70%) | Recall<br>Generous | Precision | Precision<br>Generous |
|:------|:-------|:---------------------------|:-------------------|:----------|:----------------------|
| 2D Unet Membrane Info | 0.68 | 0.76<br>0.66 | 0.78 | 0.55 | 0.65 |
| 2D-3D Unet | 0.55 | 0.74<br>0.61 | 0.75 | **0.75** | 0.78 |
| ExtendNet | **0.77** | **0.80**<br>**.72** | **0.82** | 0.42 | 0.54 |

</div>
</p>
