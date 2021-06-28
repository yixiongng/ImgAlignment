from pdf2image import *
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", required=True,
	help="path to input image that we'll align to template")
args = vars(ap.parse_args())

pages=convert_from_path(args["pdf"])
base=os.path.basename(args["pdf"])
imgname = os.path.splitext(base)[0]
for index, page in enumerate(pages):
	page.save(imgname + '_' + str(index) + '.jpg', 'JPEG')
