import PIL
import os
from PIL import Image, ImageFont, ImageDraw


def get_dir(string):
    par_dir = "./alphabet1"
    new_dir = f"{string.lower()}"
    path = os.path.join(par_dir, new_dir)
    if not os.path.exists(path):
        os.makedirs(path)


def get_image(string):
    # create an image
    # im = PIL.Image.new("1", size=(128, 128), color=1)
    # get a font
    font_list = ["arial.ttf", "ariali.ttf", "arialbd.ttf", "arialbi.ttf", "ariblk.ttf", "bahnschrift.ttf",
                 "calibril.ttf", "calibrili.ttf", "calibri.ttf", "calibrii.ttf", "calibrib.ttf", "calibriz.ttf",
                 "cambriai.ttf", "cambriab.ttf", "cambriaz.ttf", "comic.ttf", "comici.ttf", "comicbd.ttf", "comicz.ttf",
                 "consola.ttf", "consolai.ttf", "consolab.ttf", "consolaz.ttf", "constan.ttf", "constani.ttf",
                 "constanb.ttf", "constanz.ttf", "corbell.ttf", "corbelli.ttf", "corbel.ttf", "corbeli.ttf",
                 "corbelb.ttf", "corbelz.ttf", "cour.ttf", "couri.ttf", "courbd.ttf", "courbi.ttf", "framd.ttf",
                 "framdit.ttf",   "georgia.ttf", "georgiai.ttf", "georgiab.ttf", "georgiaz.ttf", "impact.ttf",
                 "javatext.ttf",  "lucon.ttf", "l_10646.ttf", "malgun.ttf", "malgunbd.ttf", "malgunsl.ttf",
                 "micross.ttf", "pala.ttf", "palai.ttf", "palab.ttf", "palabi.ttf",  "segoepr.ttf", "segoeprb.ttf",
                 "segoesc.ttf", "segoescb.ttf", "segoeuil.ttf", "seguili.ttf", "segoeuisl.ttf", "seguisli.ttf",
                 "segoeui.ttf", "segoeuii.ttf", "seguisb.ttf", "seguisbi.ttf", "segoeuib.ttf", "segoeuiz.ttf",
                 "seguibl.ttf", "seguibli.ttf", "sylfaen.ttf", "tahoma.ttf", "tahomabd.ttf", "times.ttf", "timesi.ttf",
                 "timesbd.ttf", "timesbi.ttf", "trebuc.ttf", "trebucit.ttf", "trebucbd.ttf", "trebucbi.ttf",
                 "verdana.ttf", "verdanai.ttf", "verdanab.ttf", "verdanaz.ttf"
                 ]
    for elem in font_list:
        im = PIL.Image.new("1", size=(28, 28), color=0)
        font = ImageFont.truetype(elem, 25)
    # get a drawing context
        d = ImageDraw.Draw(im)
    # draw multiline text
        d.text((14, 14), string, font=font,
               anchor='mm',
               fill=255)
        par_dir = "./alphabet1"
        new_dir = f"{string.lower()}"
        path = os.path.join(par_dir, new_dir)
        im.save(path + "\\" + f"{string.lower()}" + f"{len(os.listdir(path))}" + "-" + f"{elem}" + ".png")


letters = 'АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'
if not os.path.exists("./alphabet1"):
    os.makedirs("./alphabet1")
for letter in letters:
    get_dir(letter)
for letter in letters:
    get_image(letter)
