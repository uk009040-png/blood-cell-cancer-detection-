# Blood Cell Cancer Detection Dataset
1. Introduction

This dataset contains blood cell images and related metadata used for detecting and classifying blood cell cancers (e.g., leukemia). The images are primarily microscopy images of blood smears, and each sample is labeled according to its cancerous or non-cancerous condition.

2. Files & Directory Structure
/blood_cell_cancer_dataset/
├── images/
│   ├── cancerous/
│   │   ├── sample1.png
│   │   ├── sample2.png
│   │   └── ...
│   └── healthy/
│       ├── sample1.png
│       ├── sample2.png
│       └── ...
├── metadata.csv  
└── README.md  


images/: Directory containing all blood cell images.

cancerous/: Contains images labeled as cancerous.

healthy/: Contains images labeled as healthy (non-cancerous).

metadata.csv: A CSV file containing metadata about each image.

README.md: This file.

3. Metadata / Data Dictionary

Here is the structure of metadata.csv:

Column Name	Type	    Description
image_id	                    string	     Unique identifier of the image file.
filename	                    string	     Name of the image file (e.g., sample1.png).
label	               integer / string    Class label (e.g., 0 = healthy, 1 = cancerous).
cell_type	                     string	     Type of blood cell (if known), e.g., lymphocyte, neutrophil.
patient_id	                     string	     Identifier for the patient.
age	                    integer	     Age of the patient (in years).
gender	                       string	     Gender of the patient (e.g., Male / Female).


4. Data Collection & Processing

Collection Method:
Samples were collected by your lab / hospital name. Blood smears were prepared, stained (specify stain), and imaged under a microscope at X magnification.

Processing:

Raw microscope images were captured in .png format.

Preprocessing steps:

Resizing images to e.g., 224×224 pixels.

Normalization (e.g., scaling pixel values).

Software / Tools Used: kaggle,Github,VS studio

Python 3.x

OpenCV / PIL (for image processing)

NumPy / Pandas (for metadata handling)

(Any other tool or library used)

5. Quality Assurance

Labeling Process:
The images were manually labeled by expert hematologists / pathologists.

Validation:
A random subset of images (e.g., 10%) was cross-checked by a second expert to ensure labeling consistency.

Known Limitations:

Some images may be out of focus.

There may be class imbalance (e.g., fewer cancerous samples).

Images may have staining artifacts.

6. Usage Notes

Intended Use:
Training and evaluating machine learning / deep-learning models for blood cell cancer detection/classification.

Format:

Images are in .png format.

Metadata is in CSV.

Preprocessing Advice:
For reproducibility, you may want to apply the same preprocessing pipeline (resize, normalize) as used during model development.

7.Acknowledgments:
Thank the people / institution / funding sources that helped in data collection.blood-cell-cancer-detection-
