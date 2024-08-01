import splitfolders

# give the input/output directory path
input_folder = 'images/images'
output  = 'images/proccessed'

# give the ratio for split
splitfolders.ratio(input_folder,output,seed=42,ratio=(.8,.2))
