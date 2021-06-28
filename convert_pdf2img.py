from pdf2image import *
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("--pdf", required=True,
	help="path to input image that we'll align to template")
args = vars(ap.parse_args())

pages=convert_from_path(args["pdf"])
for page in pages:
	page.save(os.path.splitext(args["pdf"])[0] + '.jpg', 'JPEG')
