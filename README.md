# image_morphing

`conda env create --name morph --file=environment.yml`

# Notebooks
custom_vis.ipynb -> Visualize custom trained face keypoint ResNet18 model
detector_vis.ipynb -> visualize SPIGA face keypoint detection model
image_morph_vis.ipynb -> visualize image morph with user input keypoint selection feature
keypoint_detector_train.ipynb -> Training notebook for face keypoint detection

# Main
`python3 engine.py` -> takes in directory of images and morphs them into a video sequentially

# Training Data
https://www.kaggle.com/datasets/prashantarorat/facial-key-point-data

# References
For training -> https://github.com/nalbert9/Facial-Keypoint-Detection
