# Functional-Group-Detector
A work-in-progress AI model which can detect the various functional groups found within organic molecules, that are either hand drawn or image generated

- Chrome Driver is used for image scraper. You can change the directory to which the image scraper writes too
- The File Renamer can be used to rename the scraped images to 1.png-x.png where x is the number of images in the specific data set
- Models for each of the three types of functional groups being tested are avaliable, there will be support for more functional groups with coming time
- FGD_model_test will test the accuracy of the model after the model is saved in a .h5 file. The test images can be scraped off the internet using the Image Scraper
- Each DS contains a zipped file for the images, all named and labeled properly. The .csv files contain the pathway to the image, and its value (whether or not it has that functionality)
