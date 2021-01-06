# Image-classifier-model
The program learns from a pre-trained model and then come up with its own model which is is then used to classifier

NOTE:
The program can be run in a jupyter notebook by loading the Image_classifier_project.html into the3 notebook.

It can also be run from the command lie via running:
    1. python train.py image_dir (You may refer to -h on the file for help) inorder to train the model and save the checkpoints
    2. python predict.py image_path with optional params as stipulated in the docstring) in order to load a saved checkpoint and then use it it to predict input image

The program requires the flowers folder which is in the following layout:

+ flowers
   + train
     + 1
       - specific image files
     + 2 
       - specific image files
     .
     .
     .
     + 102
       - specific image files
       
   + test
     + 1
       - specific image files
     + 2 
       - specific image files
     .
     .
     .
     + 102
       - specific image files
   
   + valid
     + 1
       - specific image files
     + 2 
       - specific image files
     .
     .
     .
     + 102
       - specific image files
       
  #Happy testing and learning of the AI products @Pitson Josiah Mwakabila
