# CS231 Final Project: TikTok Effects
![Python](https://img.shields.io/badge/python-3.9-blue) \[[slide](https://docs.google.com/presentation/d/1pFMhJcswig12k3I7FHy-tDi5ue0Yb3gKHstsSjXmisg/edit?usp=sharing)\]
# Description
In this project, we use Python to implement some TikTok video effects. Intuitively, we also build a simple web demo that users could record a video and be able to apply effects just like recording a TikTok video.

You can check detail information of implemented effects in the attached slide.
# Members

<table>
  <tr>
    <th>No.</th>
    <th>Full name</th>
    <th>Student ID</th>
    <th>Gmail</th>
    <th>Github</th>
  </tr>
  <tr>
    <th>1</th>
    <th>Dinh Thang-Duong</th>
    <th>19522195</th>
    <th>19522195@gm.uit.edu.vn</th>
    <th>https://github.com/wjnwjn59</th>
  </tr>
  <tr>
    <th>2</th>
    <th>Nguyen Thuan-Duong</th>
    <th>19522312</th>
    <th>19522312@gm.uit.edu.vn</th>
    <th>https://github.com/ThuanNaN</th>
  </tr>
   <tr>
    <th>3</th>
    <th>Dinh Duc Tri-Truong</th>
    <th>19522395</th>
    <th>19522395@gm.uit.edu.vn</th>
    <th>https://github.com/TruongDinhDTri</th>
  </tr>
</table>

# Instruction
## Step 0 (Optional): Install conda
Using conda environment is recommended. You can download conda following this instruction: [link](https://conda.io/projects/conda/en/stable/user-guide/install/download.html). When conda is installed, create a conda environment and activate it using the below commands:
```
$ conda create -n tiktok_effects_env python=3.9 -y
$ conda activate tiktok_effects_env
```
## Step 1: Install required libraries  
```
$ pip install -r requirements.txt
```
## Step 2: Download pretrained model
* For pixellib library: [link](https://github.com/ayoolaolafenwa/PixelLib/releases/download/1.1/xception_pascalvoc.pb). This model should be placed in `./models/pixellib_models` directory.
## Step 3: Run the demo
```
$ python app.py
```
