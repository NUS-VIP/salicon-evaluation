# salicon-evaluation
evaluation API for salicon : a counterpart of saliency prediction as https://github.com/tylin/coco-caption

# SALICON Annotation file format:

## 1. Basic structure (same as coco other annotations):
```
{
"info" : info,
"type" : str,
"images" : [image],
"annotations" : [annotation],
"licenses" : [license],
}

info {
"year" : int,
"version" : str,
"description" : str,
"contributor" : str,
"url" : str,
"date_created" : datetime,
}

images[{
"id" : int,
"width" : int,
"height" : int,
"file_name" : str,
"license" : int,
"url" : str,
"date_captured" : datetime,
}]

licenses[{
"id" : int,
"name" : str,
"url" : str,
}]
```
## 2. Saliency Annotations:
```
annotations[{
"id" : int,
"image_id" : int,
"worker_id": int,
"fixations" : [[x0,y0],[x1,y1],...]
}]
```
# SALICON Result file format:
```
[{
"image_id" : int,
"saliency_map" : [[pointvalue-row1col1,pointvalue-row1col2,...],[pointvalue-row2col1,pointvalue-row2col2,...], ...]
}]
```
