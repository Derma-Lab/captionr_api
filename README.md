
Step 0: Add your images to the "image" directory given in the folder and set output as "output" directory. 

1. Setup a venv using:


```
uv venv caption_env

caption_env\bin\activate
```

2.  
Instlal the requirements

```
pip install -r requirements.txt
```

3.

Default command to load the api
```
python captionr.py image --output output --clip_flavor --port 8200
```