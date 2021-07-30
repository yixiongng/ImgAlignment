To execute the code, run the following command: <br />
```python image_alignment.py --template {TEMPLATE.JPG} --image scans/{ROTATED_IMAGE.JPG}``` <br />
1. Replace `{TEMPLATE.JPG}` with the template image name and replace `{ROTATED_IMAGE.JPG}` with the image name that needed to be aligned


## image-alignment

1. To execute the code, run the following command: 
  ```bash
  python3 image_alignment.py --template {TEMPLATE PATH} --image {ROTATED_IMAGE PATH}
  ```

2. Replace `{TEMPLATE PATH}`  and `{ROTATED_IMAGE PATH}` with the path of the template and rotated images path respectively e.g.  
  ```bash
    python3 image_alignment.py --template template/template.jpg --image images/images.jpg
  ```

*This source code is referring from this [website](https://www.pyimagesearch.com/2020/08/31/image-alignment-and-registration-with-opencv/).