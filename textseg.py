import numpy as np
import cv2
import math
import pytesseract
from PIL import Image
import json
import uuid
import pandas as pd
############ Utility functions ############
def decode(scores, geometry, scoreThresh):
    detections = []
    confidences = []

    ############ CHECK DIMENSIONS AND SHAPES OF geometry AND scores ############
    assert len(scores.shape) == 4, "Incorrect dimensions of scores"
    assert len(geometry.shape) == 4, "Incorrect dimensions of geometry"
    assert scores.shape[0] == 1, "Invalid dimensions of scores"
    assert geometry.shape[0] == 1, "Invalid dimensions of geometry"
    assert scores.shape[1] == 1, "Invalid dimensions of scores"
    assert geometry.shape[1] == 5, "Invalid dimensions of geometry"
    assert scores.shape[2] == geometry.shape[2], "Invalid dimensions of scores and geometry"
    assert scores.shape[3] == geometry.shape[3], "Invalid dimensions of scores and geometry"
    height = scores.shape[2]
    width = scores.shape[3]
    for y in range(0, height):

        # Extract data from scores
        scoresData = scores[0][0][y]
        x0_data = geometry[0][0][y]
        x1_data = geometry[0][1][y]
        x2_data = geometry[0][2][y]
        x3_data = geometry[0][3][y]
        anglesData = geometry[0][4][y]
        for x in range(0, width):
            score = scoresData[x]

            # If score is lower than threshold score, move to next x
            if(score < scoreThresh):
                continue

            # Calculate offset
            offsetX = x * 4.0
            offsetY = y * 4.0
            angle = anglesData[x]

            # Calculate cos and sin of angle
            cosA = math.cos(angle)
            sinA = math.sin(angle)
            h = x0_data[x] + x2_data[x]
            w = x1_data[x] + x3_data[x]

            # Calculate offset
            offset = ([offsetX + cosA * x1_data[x] + sinA * x2_data[x], offsetY - sinA * x1_data[x] + cosA * x2_data[x]])

            # Find points for rectangle
            p1 = (-sinA * h + offset[0], -cosA * h + offset[1])
            p3 = (-cosA * w + offset[0],  sinA * w + offset[1])
            center = (0.5*(p1[0]+p3[0]), 0.5*(p1[1]+p3[1]))
            detections.append((center, (w,h), -1*angle * 180.0 / math.pi))
            confidences.append(float(score))

    # Return detections and confidences
    return [detections, confidences]


def dilate_return_img(orig,mask, img_name):
    content_dict = {}
    x_list = []
    y_list = []
    w_list = []
    h_list = []
    name_list = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))

    dilate = cv2.dilate(mask, kernel, iterations=10)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts,bboxes = sort_contours(cnts)
    split_img =[]
    for c in range(len(cnts)):
        name_list.append(img_name+'_'+str(c))
        x,y,w,h = cv2.boundingRect(cnts[c])
        x,y,w,h =map(int,[x,y,w,h])
        cv2.rectangle(orig, (x, y), (x + w, y + h), (36,255,12), 2)
        split_img.append(orig[y:y+h, x:x+w])
        content_dict[c] = ocr(orig[y:y+h, x:x+w])
        x_list.append(x)
        w_list.append(w)
        y_list.append(y)
        h_list.append(h)
        df = pd.DataFrame()
        

# adding id to orig bbox
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (x+20, y+20)
        fontScale =2
        color = (36,255,12)
        thickness = 3
        # orig = cv2.putText(orig, str(c), org, font, 
        #                    fontScale, color, thickness, cv2.LINE_AA)
    df['Name'] = name_list
    df['x_value'] = x_list
    df['w_value'] = w_list
    df['y_value'] = y_list
    df['h_value'] = h_list
    #df.to_csv('seg_diff_csv/'+img_name+'.csv')
    return(orig, dilate, content_dict,df, split_img)


def sort_contours(cnts):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][1], reverse=False))
    return (cnts, boundingBoxes)


def get_text_seg(frame, image):

    img_name = image

    confThreshold = 0.8
    nmsThreshold = 0.7
    inpWidth = 2048
    inpHeight = 2048
    model = "frozen_east_text_detection.pb"

    # Load network
    net = cv2.dnn.readNet(model)
    net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
    

    
    outputLayers = []
    outputLayers.append("feature_fusion/Conv_7/Sigmoid")
    outputLayers.append("feature_fusion/concat_3")
    height_ = frame.shape[0]
    width_ = frame.shape[1]
    rW = width_ / float(inpWidth)
    rH = height_ / float(inpHeight)


    mask = np.zeros(frame.shape[:2], np.uint8)
    # Create a 4D blob from frame.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (inpWidth, inpHeight), (123.68, 116.78, 103.94), True, False)

    # Run the model
    net.setInput(blob)
    output = net.forward(outputLayers)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())

    # Get scores and geometry
    scores = output[0]
    geometry = output[1]
    [boxes, confidences] = decode(scores, geometry, confThreshold)
    # Apply NMS
    indices = cv2.dnn.NMSBoxesRotated(boxes, confidences, confThreshold,nmsThreshold)
    for i in indices:
        # get 4 corners of the rotated rect
        vertices = cv2.boxPoints(boxes[i[0]])
        # scale the bounding box coordinates based on the respective ratios
        x,y=[],[]
        for j in range(4):
            vertices[j][0] *= rW
            vertices[j][1] *= rH
        for j in range(4):
            p1 = (vertices[j][0], vertices[j][1])
            p2 = (vertices[(j + 1) % 4][0], vertices[(j + 1) % 4][1])
            x.append(int(vertices[j][0]))
            x.append(int(vertices[(j + 1) % 4][0]))
            y.append(int(vertices[j][1]))
            y.append(int(vertices[(j + 1) % 4][1]))
            # cv2.line(frame, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
            # cv2.putText(frame, "{:.3f}".format(confidences[i[0]]), (vertices[0][0], vertices[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.rectangle(mask, (min(x),min(y)), (max(x),max(y)),  (255, 0, 0),-1)

    output_img,dilate, content_dict, df, split_img=dilate_return_img(frame,mask, img_name)
    return(output_img,label,dilate, content_dict, df, split_img)
    #return(output_img,label,dilate, mask)


def image2text(image):
    return pytesseract.image_to_string(image)


def get_text_image(text_data:dict)->np.ndarray:

    """
    Input:
    -------
        data: A dictionary containing the data to be placed in image

    Description:
    ------------
        We use cv2 "putext" to add text to image of (2048,2048) size

    Output:
    --------
        text_image: Image containing text as numpy array   
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    org = (10, 0)
    font_scale = 1.2
    color = (0, 0, 0)
    thickness = 2

    
    text_image = Image.new('RGB',(2048,2048), (255,255,255))
    text_image=np.array(text_image)



    for k,v in text_data.items():

        if(org[1]+80>2000):
            org=(1000,0)
        org=(org[0],org[1]+80)
        if(type(v)==float):
            text_image = cv2.putText(text_image,f"{k} :{v:.3f}", org, font, font_scale, color, thickness, cv2.LINE_AA)
        else:
            v=str(v)
            text_image = cv2.putText(text_image,f"{k} :{v}", org, font, font_scale, color, thickness, cv2.LINE_AA)
            
    return(text_image)


def ocr(orig):
    
    # c = count
    # image = image_name
    text_image= cv2.resize(get_text_image({"text":image2text(orig)}), (512, 512))
    
    text_image_1=image2text(orig)
    
    # with open ('txt_converted/'+image.split('.')[0]+'.txt','a+') as f:
        
    #     f.write(text_image1)
    
    
    overlay_image= cv2.resize(orig, (512, 512))
    combined_image = cv2.hconcat([overlay_image, text_image])
    # json_converted =  json.dumps(dic)
    # with open ('txt_converted/'+image+'.json','a+') as f:
    #     f.write(json_converted)
    #     f.write(',')
    #cv2.imwrite(f"ocr_single_col/{str(uuid.uuid4())}.jpeg",combined_image)
    return(text_image_1)