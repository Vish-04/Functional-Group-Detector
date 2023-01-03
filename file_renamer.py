import os

# Change the current working directory to the folder with the images
os.chdir('C:\\Py Projects\\Machine Learning\\Functional Group Detector\\Alkene DS\\imgs')

# Get a list of all the files in the folder
files = os.listdir()

# Rename the files
# for i, file in enumerate(files):
#   os.rename(file, str(i) + '.png')

i = 508
while i in range(508, len(files)):
  try:
    os.rename(files[i], str(i) + ".png")
  except:
    continue
  i = i+1