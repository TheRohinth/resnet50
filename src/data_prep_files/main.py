import os
import json
import augment

paths = json.loads(open('paths.json',).read())                            # Get paths from paths.json
root_path = paths["train_data_root_folder"]                               # Get root path
images = os.listdir(root_path)                                            # Get all image file names in the root path
images = images[:2]                                                       # Only get the first two images
#print(images,"\n\n\n")
for image in images:                                                      # Loop through all images
    image = os.path.splitext(image)[0]
    image_path = os.path.join(root_path, image)                           # Get the image path

    _, success               = augment.flip(image_path, False)       # Vertical flip
    if not success:
        print("Vertical flip failed for image: ", image_path)

    _, success               = augment.flip(image_path, True)        # Horizontal flip
    if not success:
        print("Horizontal flip failed for image: ", image_path)

    _, success = augment.rotate_90_clockwise(image_path, "_2")       # Rotate 90 degrees clockwise
    if success:
        _, success = augment.rotate_90_clockwise(image_path, "_3")       # Rotate 180 degrees clockwise
        if success:
            _, success = augment.rotate_90_clockwise(image_path, "_4")       # Rotate 270 degrees clockwise
            if not success:
                print("Error: 270 - Failure")
        else:
            print("Error: 180 - Failure => 270 Aborted")
    else:
        print("Error: 90 - Failure => 180 Aborted => 270 Aborted")