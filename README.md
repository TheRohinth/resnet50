Project - Retinal Disease Classification.

Concept used - Deep Learning

--------------------------------------------------------------

Dataset used - "https://www.kaggle.com/c/vietai-advance-retinal-disease-detection-2020/overview"

classes list :-
    - opacity
    - diabetic retinopathy
    - glaucoma
    - macular edema
    - macular degeneration
    - retinal vascular occlusion
    - normal

values (boolean)

--------------------------------------------------------------

Data Augmentation :-

Library used :- 
    - cv2   -> to process the image.
    - json  -> to parse the path of root folder.
    - os    -> to list the files of the root folder.

Main.py -> "src/data_prep_files/main.py"

Util Files :-
    - augment.py    
            Augmentations Defined :-
                    1) Horizontal Flip
                    2) Vertical Flip
                    3) Rotate_90_Clockwise

--------------------------------------------------------------

Model selected - ResNet-50

ResNet50 Architecture - "src/ResNet50_Architecture_Design/ResNet50_architecture.png"

ResNet50 Main file - "src/DL_model_files/init_resnet_50.py"

Util Files :-
    - blocks.py
            Methods :-
                1) Convolutional_block
                2) Identity_block


--------------------------------------------------------------

Our main metric of model evaluation is :-
    1) F1 score
    2) Kappa score

![F1 score formula](https://miro.medium.com/max/1400/1*wUdjcIb9J9Bq6f2GvX1jSA.png)
![Kappa score formula](https://www.researchgate.net/profile/Edward-Shortliffe/publication/220387601/figure/fig2/AS:668992054247429@1536511543431/The-kappa-coefficient-of-agreement-This-equation-measures-the-fraction-of-beyondchance.png)