1. Save h5 file of model
2. Change file path in Deepface/deepface/basemodels/Resnet34_finetune.py at line 50.
```
model = keras.models.load_model("PATH to h5 file")
```

