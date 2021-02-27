# safe-driving-classification

## Installation
Enter virtual environment:
```
virtualenv venv
source venv/bin/activate
```
Install requirements:
```
pip3 -r requirements.txt
```

## Training
### From scratch:
Run the train script:
```
python3 train.py DIRECTORY_TO_IMAGE_FILES DIRECTORY_TO_SAVE_CHECKPOINT_FILE
```

### From pretrained model:
Download ckpt file from releases and run the train script:
```
python3 train.py DIRECTORY_TO_IMAGE_FILES DIRECTORY_OF_CHECKPOINT_FILE
```
## Dataset:
The dataset used for this project can be found in the releases. If you want to train with your own dataset, ensure the following file structure:

	.
	├── images
	│	├── safe
	│	|	├── image1.png
	│	|	├── image2.png
	|	|	└── ...
	│	├── unsafe
	│	|	├── image1.png
	│	|	├── image2.png
	|	|	└── ...