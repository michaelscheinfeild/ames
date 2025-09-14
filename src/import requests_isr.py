import requests

export_url = "https://data-israeldata.opendata.arcgis.com/"
params = {
    "bbox": "34.0,29.0,36.0,33.0",   # lon,lat of SW & NE corners
    "bboxSR": "4326",
    "size": "10000,10000",           # width,height in px
    "format": "tiff",
    "f": "image"
}

r = requests.get(export_url, params=params, stream=True)
with open("israel_ortho_10000.tif", "wb") as f:
    for chunk in r.iter_content(1024 * 1024):
        f.write(chunk)