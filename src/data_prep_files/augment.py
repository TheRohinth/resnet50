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
            image_path :-
                Type        :-  String
                Description :-  Path of the flipped image stored to the disk.
            listen_success :-
                Type        :-  bool
                Description :-  Boolean flag to denote the success of the operation.

    '''
    if(flip_type):                                   
        image = cv2.flip(cv2.imread(image_path+".jpg"), 1)     # flipcode > 0: flip horizontally
        image_path += "_1.jpg"                              # flip horizontally -> image name + "_1"
    else:
        image = cv2.flip(cv2.imread(image_path+".jpg"), 0)     # flipcode=0: flip vertically
        image_path += "_0.jpg"                              # flip vertically -> image name + "_0"
    #print(image_path)
    listen_success = cv2.imwrite(image_path, image)     # save image
    return image_path, listen_success                   # return path of the new image with boolean of success or failure


def rotate_90_clockwise(image_path: str, file_id: str):
    '''
        Function name   :- rotate_90_clockwise
        Description     :- This function is used to rotate the input image 90 degrees clockwise and store it to the disk.

        Parameters :-
            image_path :-
                Type        :-  String
                Description :-  Full path to the image + image name + image extension.
            file_id     :-
                Type        :-  str
                Accepted values :-  "_2" or "_3" or "_4"
                Description :-  Unique id string of the image.

        Returns :-
            image_path :-
                Type        :-  String
                Description :-  Path of the rotated image stored to the disk.
            listen_success :-
                Type        :-  bool
                Description :-  Boolean flag to denote the success of the operation.
    '''
    if(file_id == "_3"):                            # if file_id is "_3" pick the previous image file with file name + "_2" + ".jpg"
        old_image_path = image_path+"_2.jpg"
    elif(file_id == "_4"):                          # if file_id is "_4" pick the previous image file with file name + "_3" + ".jpg"
        old_image_path = image_path+"_3.jpg"
    else:                                           # if file_id is "_2" pick the previous image file which is original file
        old_image_path = image_path+".jpg"
    #print(old_image_path)                                            
    listen_success = cv2.imwrite(
            image_path+file_id+".jpg",
            cv2.rotate(
                cv2.imread(old_image_path),
                cv2.cv2.ROTATE_90_CLOCKWISE
            )
        )          # cv2.imread -> read the image   &  cv2.rotate -> rotate the image  +  ROTATE_90_CLOCKWISE  &  cv2.imwrite -> save the image
    return image_path, listen_success                # returns the path of the new image with boolean of success or failure