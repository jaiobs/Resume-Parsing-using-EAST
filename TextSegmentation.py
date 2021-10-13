#!/usr/bin/env python
# coding: utf-8

# ## Import Necessary libraries

# In[1]:


import textseg as ts
from PyPDF2 import PdfFileReader
from pdf2image import convert_from_path
import cv2
import json
import pandas as pd
import numpy as np
import glob
import os
import pytesseract
import math
path_to_write = "TesseractDemo/Output/"


# ## The below function is to convert your pdf to image data

# In[2]:


def convert_pdf_to_image(filepath,img_path_to_save):
    try:
        fileName = filepath.split("/")[-1].replace(".pdf","")
        pages = convert_from_path(filepath, 350)
        i = 1
        for page in pages:
            image_name = img_path_to_save+fileName+"Page_" + str(i) + ".png"  
            page.save(image_name, "JPEG")
            i = i+1
        return {"status":200,"response":"PDF Converted to image sucessfully","fileName":fileName}
    except Exception as e:
        return {"status":400,"response":str(e)}


# ## get the list of documents you want to pass as an input

# In[3]:


documents = glob.glob("TesseractDemo/*.pdf")
documents = documents[:6]


# ### The below function is used get the text present in a image

# In[4]:


def text_from_tesseract(output_img):
    text = str(((pytesseract.image_to_string(output_img))))
    return text


# ### This function is the core function to process each pdf and store the resultant output using EAST-Text detection Model

# In[5]:


data = pd.DataFrame()
final_name_list=[]
final_text_opencv=[]
final_text_tessaract=[]
for i in documents:
    pdf = PdfFileReader(open(i,'rb'))
    fname = i.split('/')[-1]
    print(pdf.getNumPages())
    images = convert_from_path(i)
    resumes_img=[]
    for j in range(len(images)):
         # Save pages as images in the pdf
        images[j].save(path_to_write+fname.split('.')[0]+'_'+ str(j) +'.jpg', 'JPEG')
        resumes_img.append(path_to_write+fname.split('.')[0]+'_'+ str(j) +'.jpg')
    name_list = fname.split('.')[0]+'_' +'.jpg'
    text_opencv=[]
    text_tessaract=[]
    for i in resumes_img:
        frame=cv2.imread(i)
        os.remove(i)
        img = i.split("/")[2]

        output_img,label,dilate, c_dict,df1, split_img=ts.get_text_seg(frame, img)
        cv2.imwrite(path_to_write+img.split('.')[0]+".png",output_img)
        for i in range(len(split_img)):
             cv2.imwrite(path_to_write+img.split('.')[0]+str(i)+".png", split_img[i])
        
        text_opencv.append(c_dict)
        text_tessaract+=text_from_tesseract(output_img)
        tesseract_str = ''.join(text_tessaract)
    
    final_name_list.append(name_list)
    final_text_opencv.append(text_opencv)
    final_text_tessaract.append(tesseract_str)   


# ### Since we have passed only one document we are looking at the fisrt index in a list

# In[6]:


final_text_opencv[0]

