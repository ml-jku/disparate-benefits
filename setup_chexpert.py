import os
import glob
from tqdm import tqdm

from PIL import Image

from source.constants import CHEXPERT_PATH

"""
The following directory uses the contents of 
https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2 
which containts train and val set with labels. 
Furthermore, the test set with labels is obtained through 
https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c. 
Finally, the CHEXPERT_DEMO file is obtained from 
https://stanfordaimi.azurewebsites.net/datasets/192ada7c-4d43-466e-b8bb-b81992bb80cf. 
All downloads require to register at Stanford AIMI. 
Total download size is about 500 GB and more than a terrabyte after decompression. 
Downsizing as done here substantially reduces the size.

A little bit of moving things around was necessary. 
I moved / merged all folders I obtained after decompression into 'val' and 'train' respectively.
"""

#! This needs to be modified for the actual path
SOURCE_PATH = "/system/user/publicdata/chexpertchestxrays-u20210408"

#! = True if need to move images to local SSD
move_images = True

# make sure directory to copy data to exists
os.makedirs(CHEXPERT_PATH, exist_ok=True)


########################
# Cleaning annotations #
########################

# copy metadata to the correct location
os.system(f"cp {SOURCE_PATH}/CHEXPERT_DEMO.xlsx {CHEXPERT_PATH}/CHEXPERT_DEMO.xlsx")
# there are three different label options for train, we use the latest (best) one
os.system(f"cp {SOURCE_PATH}/train_visualCheXbert.csv {CHEXPERT_PATH}/train.csv")
os.system(f"cp {SOURCE_PATH}/valid.csv {CHEXPERT_PATH}/valid.csv")
os.system(f"cp {SOURCE_PATH}/test.csv {CHEXPERT_PATH}/test.csv")

#########################
# Moving images to disk #
#########################

if move_images:

    for split in ["train", "valid", "test"]:

        target_dir = CHEXPERT_PATH + f"/{split}"
        source_dir = SOURCE_PATH + f"/{split}"

        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)

        print(f"searching for images in the {split} source directory...")

        # Get a list of all image files in the source directory
        image_files = glob.glob(source_dir + "/**/*.jpg", recursive=True)

        print("done")

        # Copy and resize each image file to the target directory
        for image_file in tqdm(image_files):
            # Get the relative path of the image file
            relative_path = os.path.relpath(image_file, source_dir)
            
            # Get the target path for the image file
            target_path = os.path.join(target_dir, relative_path)
            
            # Create the target directory if it doesn't exist
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            
            # Open the image file
            image = Image.open(image_file)
            
            # Resize the image to 224x224
            resized_image = image.resize((224, 224))
            
            # Save the resized image to the target path
            resized_image.save(target_path)

        # Print a message when the copying and resizing is complete
        print("Copying and resizing images complete.")
