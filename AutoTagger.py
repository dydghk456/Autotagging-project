import json, os, re, cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import pandas as pd
from google.colab.patches import cv2_imshow

from db import DB
pd.set_option('display.max_colwidth', None)


class AutoTagger():

    def __init__(self, deepface, vector_path, json_path, people_db_path):
        self.model = deepface
        self.vector_path = vector_path
        self.json_path = json_path
        self.font_size = 12
        self.font = ImageFont.truetype("/content/drive/MyDrive/chang/fonts/NanumGothic.ttf", self.font_size)
        self.people_db = DB(people_db_path)

    def tag(self, target_path, similarity = "cosine", print_face=True, write_json=True, enforce_detection=True):
        #calculate similarity between images in db
        similarity = self.model.find(img_path = target_path, db_path = self.vector_path, distance_metric = similarity, silent=True,enforce_detection=enforce_detection)
        #count the number of faces in image
        target_num = len(similarity)
        results = pd.DataFrame()
        #for each face in image, append found person in db
        for i in range(target_num):
            #there is no matched person in db
            if len(similarity[i]) < 1 :
                result = pd.DataFrame(data={'identity': 'not_matched', 'source_x':0, 'source_y':0, 'source_w':0, 'source_h':0, f"VGG_FACE_{similarity}": -1, 'target_img':target_path, 'matched_uid':-1, 'mathced_name':'none'}, index=[0])
            similarity[i] = similarity[i].iloc[0:1]
            similarity[i]['target_img'] = target_path
            similarity[i]['matched_uid'] = similarity[i]['identity'].str.split('uid=|[_]|[0-9]+.jpg').apply(lambda x : x[3])
            similarity[i]['matched_name'] = similarity[i]['identity'].str.split('uid=|[_]|[0-9]+.jpg').apply(lambda x: x[-2])
            results = results.append(similarity[i].iloc[0:1])
        
        #if print_face is true, show image
        if(print_face):
            self.print_face(results, target_path)
        
        #if write_json is true, write json file
        if(write_json):
            self.write_json(results)

        return results
    
    def print_face(self, results, target_path):
        #load image
        img_pil = Image.open(target_path)
        draw = ImageDraw.Draw(img_pil)
        #for each face in image, write name of recognized person
        for idx, row, in results.iterrows():
            draw.text((row['source_x'], row['source_y']-self.font_size-3), row['matched_name'], (255,255,0), font = self.font)
        img = np.array(img_pil)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        #draw bounding box of face
        cv2.rectangle(img, (row['source_x'],row['source_y']), (row['source_x']+row['source_w'],row['source_y']+row['source_h']), (0,0,255), 2)
        cv2_imshow(img)

    def write_json(self, results):
        result_dict = {}
        for idx, row, in results.iterrows():
            result_dict[row['matched_uid']] = {"name":row['matched_name'], "bbox":[row['source_x'],row['source_y'],row['source_x']+row['source_w'],row['source_y']+row['source_h']], "valid":True}
        
        with open(self.json_path+"/faces.json", "w", encoding="utf-8") as file_path:
            json.dump(result_dict, file_path, ensure_ascii=False, indent="\t")

    def insert_person(self, name):
        return self.people_db.insert_person(name)
    
    def print_people(self):
        self.people_db.show_people()

    def check_person_exists(self, name):
        self.people_db.check_exists(name)

    def calc_metrics(self, target_path, similarity = "cosine"):
        img_path = os.listdir(target_path)
        total_results = pd.DataFrame()
        for img in img_path:
            
            #if file is not jpg image, or there are multiple or no faces, skip prediction
            if os.path.isfile(target_path+"/"+img) and img.lower().endswith(('.jpg','.jpeg')):
                faces = self.model.extract_faces(img_path=target_path+"/"+img, enforce_detection=False)
                if len(faces) != 1 or faces[0]['confidence']==0:
                    continue
                else:
                    #find matched person in db, and write whether matching is correct in 'correct' column in results
                    result = self.tag(target_path+"/"+img, similarity=similarity, print_face=False, write_json=False, enforce_detection=False)
                    parsed_img = re.split('uid=|[_]|[0-9]+.jpg', img)
                    result['target_uid'] = parsed_img[1]
                    result['target_name'] = parsed_img[-2]
                    result['correct'] = (result['target_uid'] == result['matched_uid'])
                    total_results = total_results.append(result, ignore_index = True)
      
        metrics = {}
        #calculate precision
        metrics['precision'] = total_results['correct'].sum()/len(total_results)
        return metrics, total_results
    