# Adaptive-Photo-Mosaic-GUI
Make mosaic images 

![mosaic_1755200205](https://github.com/user-attachments/assets/ff37eb60-82a2-4d46-89a5-6253057d8d40)

# First time
    Download source (click "Code" and "Download zip"), extract the zip, install requirements, open adaptive_photo_mosaic_gui.py

  OR

    Use the prebuilt version built with PyInstaller - open the .exe file.

# Requirements
  If you haven't already, install Python. On Windows it's relatively simple - open cmd as administrator and type in: 

    winget install python

      (Hopefully it should automatically add Python to PATH. Try the command below - if it fails, look up "Add Python to PATH on Windows")


  Before opening adaptive_photo_mosaic_gui.py, you have to open cmd as administrator and type in:

    pip install pillow numpy

# Usage
  1. Download a variety of images (the more colours the better) and put them inside a folder.

    You can use e.g. https://data.caltech.edu/records/mzrjq-6wc02 (download and extract the caltech-101.zip file) inside which you can find 101_ObjectCategories.tar.gz - extract that and you have a variety of different sample folders.

  2. Select the target image inside the app. Be sure its dimensions are large enough.

  3. Select the sample folder ("Source folder") with the images you downloaded.

  4. Click "Scan source folder ..."

  5. Select the folder where you want the finished mosaic image ("Output folder")

  6. Click "Generate image"

# Other options
  You should try all the options you find (like the "Half mosaic" with the slider) and see for yourself what they do :P
