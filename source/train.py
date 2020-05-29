import os

baseDir = os.path.dirname(os.path.abspath(__file__))

imageDir = os.path.join(baseDir, "images")


# sprehajamo se znotraj zbirke images
# in preverjamo vsako datoteko vanjo
for root, dirs, files in os.walk(imageDir):
    for file in files:
        if file.endswith("jpg") or file.endswith("jpg"):
            path = os.path.join(root, file) # dobimo pot do slike, ki ima končnico .jpg ali .png
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower() #dobimo direktorij od slike 
            print(label, "Path:", path)
