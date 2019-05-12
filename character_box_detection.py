'''
Taken almost without modification from Kanan Vyas's tutorial on
box detection with OpenCV. https://github.com/KananVyas/BoxDetection
'''

import cv2
import numpy as np

import io
import os

# export GOOGLE_APPLICATION_CREDENTIALS=hanzi-ocr-6187fe679c36.json

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud import translate
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()
translate_client = translate.Client()

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def detect_text(path):
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
        print(path)

    image = vision.types.Image(content=content)

    response = client.document_text_detection(
        image=image,
        image_context={"language_hints": ["zh"]})

    annotations = response.text_annotations

    if len(annotations) > 0:
        text = annotations[0].description
    else:
        text = ''
    print('Extracted text {} from image ({} chars).'.format(text, len(text)))

    detect_language_response = translate_client.detect_language(text)
    src_lang = detect_language_response['language']
    print('Detected language {} for text {}.'.format(src_lang, text))

    # # if len(annotations) > 0:
    # #     text = annotations[0].description
    # # else:
    # #     text = ''
    # text = ''
    # for t in annotations:
    #     text = t.description
    # print('Extracted text {} from image ({} chars).'.format(text, len(text)))

    # texts = response.text_annotations
    # string = ''
    #
    # if (len(texts) > 0):
    #     string =  texts[0].description
    #
    # for text in texts:
    #     string+=' ' + text.description

    # return '$$ ' + string + ' $$'

def box_extraction(img_for_box_extraction_path, cropped_dir_path):
    img = cv2.imread(img_for_box_extraction_path, 0)  # Read the image
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    # cv2.imwrite("./test/image_bin.jpg",img_bin)

    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//40

    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    # cv2.imwrite("./test/verticle_lines.jpg",verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    # cv2.imwrite("./test/horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    # cv2.imwrite("./test/img_final_bin.jpg", img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all the contours by left to right.
    (contours, boundingBoxes) = sort_contours(contours, method="left-to-right")
        # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    idx = 0
    # Need these threshold to get just each individual character square.
    # Have not yet tested with just 口, only images with kou radical
    square_threshold = 2.0

    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        # If the box is essentially a square, then only save it as a box in "characters/" folder.
        if (abs(w - h) < square_threshold):
            idx += 1
            new_img = img[y:y+h, x:x+w]
            cv2.imwrite(cropped_dir_path + str(idx) + '.png', new_img)
            text = detect_text(cropped_dir_path + str(idx) + '.png')
            print(text)

    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    # cv2.imwrite("./temp/img_contour.jpg", img)

box_extraction("en_test-multi-blank.png", "./characters/")
