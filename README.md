# 2021DLCV_Final_Nodule_Detedction

### Objective
Train a neural network that identifies locations of the possible nodules that are larger than 3 mms by utilizing data annotated by several experienced radiologists.

We divided the objective into two main tasks:
- Segmentation Part: use U-net to segment and localize the candidate voxels of lung nodules.
- Classification Part:  use a 3D CNN model to classify the probability of nodule of each candidate voxel.



### Dataset
- seg-lungs-Luna16 (partial): zip file which contain a sample of all CT images

- annotations.csv: csv file that contains the annotations. Each line holds the SeriesInstanceUID of the scan, the x, y, and z position of each finding in world coordinates; and the corresponding diameter in mm.


