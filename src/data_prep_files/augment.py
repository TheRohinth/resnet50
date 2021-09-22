import cv2

def flip(image_path: str, flip_type: bool):
    ''' 
        Function name :- flip
        Description :- This function is used to flip the image horizontally or vertically

        Parameters :-
            image_path :- 
                Type        :-  String
                Description :-  Full path to the image + image name + image extension.

            flip_type  :-  
                Type            :-  bool
                Accepted values :-  False -> vertical, True -> horizontal
                Description     :-  boolean flag to denote the type of flip to be made to the input image.

        Returns :-
            listen_success :-
                Type        :-  bool
                Description :-  Boolean flag to denote the success of the operation.

    '''
    if(flip_type):                                   
        image = cv2.flip(cv2.imread(image_path+".jpg"), 1)      # flipcode > 0: flip horizontally
        image_path += "_1.jpg"                                  # flip horizontally -> image name + "_1"
    else:
        image = cv2.flip(cv2.imread(image_path+".jpg"), 0)      # flipcode=0: flip vertically
        image_path += "_0.jpg"                                  # flip vertically -> image name + "_0"
    #print(image_path)
    listen_success = cv2.imwrite(image_path, image)             # save image
    return listen_success                                       # returns a boolean of success or failure


def rotate_90_clockwise(image_path: str):
    '''
        Function name   :- rotate_90_clockwise
        Description     :- This function is used to rotate the input image 90 degrees clockwise and store it to the disk.

        Parameters :-
            image_path :-
                Type        :-  String
                Description :-  Full path to the image + image name + image extension.

        Returns :-
            listen_success :-
                Type        :-  bool
                Description :-  Boolean flag to denote the success of the operation.
    '''                                           
    listen_success = cv2.imwrite(
            image_path+"_2.jpg",
            cv2.rotate(
                cv2.imread(image_path),
                cv2.cv2.ROTATE_90_CLOCKWISE
            )
        )          # cv2.imread -> read the image   &  cv2.rotate -> rotate the image  +  ROTATE_90_CLOCKWISE  &  cv2.imwrite -> save the image
    return listen_success                # returns a boolean of success or failure

def rotate_180_clockwise(image_path: str):
    '''
        Function name   :- rotate_180_clockwise
        Description     :- This function is used to rotate the input image 180 degrees clockwise and store it to the disk.

        Parameters :-
            image_path :-
                Type        :-  String
                Description :-  Full path to the image + image name + image extension.

        Returns :-
            listen_success :-
                Type        :-  bool
                Description :-  Boolean flag to denote the success of the operation.
    '''                                           
    listen_success = cv2.imwrite(
            image_path+"_3.jpg",
            cv2.rotate(
                cv2.imread(image_path),
                cv2.ROTATE_180
            )
        )          # cv2.imread -> read the image   &  cv2.rotate -> rotate the image  +  ROTATE_180_CLOCKWISE  &  cv2.imwrite -> save the image
    return listen_success                # returns a boolean of success or failure

def rotate_90_counter_clockwise(image_path: str):
    '''
        Function name   :- rotate_90_counter_clockwise (wiz 270 deg clockwise)
        Description     :- This function is used to rotate the input image 90 degrees counter clockwise and store it to the disk.

        Parameters :-
            image_path :-
                Type        :-  String
                Description :-  Full path to the image + image name + image extension.

        Returns :-
            listen_success :-
                Type        :-  bool
                Description :-  Boolean flag to denote the success of the operation.
    '''                                           
    listen_success = cv2.imwrite(
            image_path+"_4.jpg",
            cv2.rotate(
                cv2.imread(image_path),
                cv2.ROTATE_90_COUNTERCLOCKWISE
            )
        )          # cv2.imread -> read the image   &  cv2.rotate -> rotate the image  +  ROTATE_180_CLOCKWISE  &  cv2.imwrite -> save the image
    return listen_success                # returns a boolean of success or failure