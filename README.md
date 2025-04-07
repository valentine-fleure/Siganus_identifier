# Siganus_identifier

This repository contains code that can be use reproduce results of the paper Fleuré et al. (2024) - Automated identification of invasive rabbitfishes in underwater images from the Mediterranean Sea submitted in Aquatic Conservation (https://doi.org/10.1002/aqc.4073).

## Install

Before using the code in this repository make sure you have the necessary libraries. You can install them as follows using pip 


```bash
pip install -r requirements.txt
```

## Usage

To use the identification model, run the following line of code, entering the path to the folder containing the images to be identified.

```bash
python main.py --path path/to/img_dir
```

A res.csv file will be added to the images folder. This file contains the name of the images, the inferred class, the image surface and the confidence score associated with the identification.

Images in the [:file\_folder: **img_test**](img_test/) folder can be used for demonstration purposes. These images come from the paper test database.