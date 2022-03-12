import urllib.request
import zipfile
from matplotlib import font_manager

urllib.request.urlretrieve("https://www.font-police.com/classique/sans-serif/arial.ttf", "FONTS/arial.ttf")
url = "https://fr.ffonts.net/Lexend-Deca-Regular.font.zip"
urllib.request.urlretrieve(url, "FONTS/lexend_deca.zip")

with zipfile.ZipFile("FONTS/lexend_deca.zip", 'r') as zip_file:
    with open("FONTS/lexend_deca.ttf", 'wb') as f:
        f.write(zip_file.read('lexenddeca/LexendDeca-Regular.ttf'))

font_dirs = ["FONTS"]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
