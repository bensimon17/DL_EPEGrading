
# Automated Detection and Grading of Extraprostatic Extension of Prostate Cancer at MRI via Cascaded Deep Learning and Random Forest Classification
[Benjamin D. Simon](https://www.linkedin.com/in/benjamin-dabora-simon/), Katie M. Merriman, Stephanie A. Harmon, Jesse Tetreault, Enis C. Yilmaz, Zoë Blake, Maria J. Merino, Julie Y. An, Jamie Marko, Yan Mee Law, Sandeep Gurram, Bradford J. Wood, Peter L. Choyke, Peter A. Pinto, Baris Turkbey

[read paper here](https://doi.org/10.1016/j.acra.2024.04.011)

## Abstract
Extraprostatic extension (EPE) is well established as a significant predictor of prostate cancer aggression and recurrence. Accurate EPE assessment prior to radical prostatectomy can impact surgical approach. We aimed to utilize a deep learning-based AI workflow for automated EPE grading from prostate T2W MRI, ADC map, and High B DWI. An expert genitourinary radiologist conducted prospective clinical assessments of MRI scans for 634 patients and assigned risk for EPE using a grading technique. The training set and held-out independent test set consisted of 507 patients and 127 patients, respectively. Existing deep-learning AI models for prostate organ and lesion segmentation were leveraged to extract area and distance features for random forest classification models. Model performance was evaluated using balanced accuracy, ROC AUCs for each EPE grade, as well as sensitivity, specificity, and accuracy compared to EPE on histopathology. A balanced accuracy score of .390 ± 0.078 was achieved using a lesion detection probability threshold of 0.45 and distance features. Using the test set, ROC AUCs for AI-assigned EPE grades 0–3 were 0.70, 0.65, 0.68, and 0.55 respectively. When using EPE≥ 1 as the threshold for positive EPE, the model achieved a sensitivity of 0.67, specificity of 0.73, and accuracy of 0.72 compared to radiologist sensitivity of 0.81, specificity of 0.62, and accuracy of 0.66 using histopathology as the ground truth. Our AI workflow for assigning imaging-based EPE grades achieves an accuracy for predicting histologic EPE approaching that of physicians. This automated workflow has the potential to enhance physician decision-making for assessing the risk of EPE in patients undergoing treatment for prostate cancer due to its consistency and automation.

## Instructions for Reproducing Automated EPE Detection
1. Follow the instructions for previously developed [prostate cancer deep learning lesion detection](https://github.com/Project-MONAI/research-contributions/tree/main/prostate-mri-lesion-seg).
2. Use requirements.txt to set up a conda environemnt (conda create --name <env> --file requirements.txt).
3. Run ContourVariance_wReduction.py to create variance masks.
4. Run AutomatedEPE_dataCollection_wSorting.py to calculated radiomics features and save them.
5. Run EPE_RandomForest.py to train and test the features and model.

Please contact benjamin.simon@nih.gov with any inquiries regarding running this code. Please cite [the paper](https://doi.org/10.1016/j.acra.2024.04.011) when using any component from this repository.

The content of this repositiory its associated publication does not necessarily reflect the views or policies of the Department of Health and Human Services, nor does mention of trade names, commercial products, or organizations imply endorsement by the U.S. Government. 