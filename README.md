AutoTagging using Deep Learning
===============================

How to use 
----------
1. Install deepface
```
$ cd Deepface
$ pip install -e .
```
2. Download pretrained model and change model path
> Change file path in Deepface/deepface/basemodels/Resnet34_finetune.py at line 50.
>```
>model = keras.models.load_model("PATH to h5 file")
>```

3. Import DeepFace and create Autotagger
```python
autotagger = Autotagger(DeepFace, vector_path, json_path,
			people_db_path, model_name)
```
4. Use tag function of Autotagger to tag people in image.
```python
autotagger.tag(target_path="path_to_image", similarity="eucliedan_l2", k)
```
similarity means distance metric and default is cosine
k means value of k in K-Nearest Neighbor and shows good results when k=8.


References
----------
* DeepFace: <https://github.com/serengil/deepface>

