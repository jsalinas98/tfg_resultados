"""
Pequeño script para guardar los clips de los momentos concretos de videos de youtube
de forma sencilla y rápida.
"""

#importar librerias necesarias
import argparse
import pytube
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from os import remove

#construir los argumentos necesarios para el programa
ap = argparse.ArgumentParser()
ap.add_argument("-l", "--link", required=True,
	help="link del video")
ap.add_argument("-n", "--name", required=True,
	help="nombre de destino del video")
ap.add_argument("-r", "--resolution", type=str, default="720p",
	help="resolucion de descarga")
ap.add_argument("-i", "--inicio", required=True, type=int,
	help="segundo inicial del video")
ap.add_argument("-f", "--final", required=True, type=int,
	help="segundo final del video")
args = vars(ap.parse_args())

#proceso de descargar del video
print("[INFO] Descargando video de YouTube...")
yt = pytube.YouTube(args["link"])
try:
	yt.streams.get_by_resolution(args["resolution"]).download()	#Lo guarda en el mismo directorio que el programa, con yt.title tengo el titulo
#Lanzamos una excepcion puesto que con algunos videos falla obtener el video filtrando la resolucion
except:
	print("[INFO] Descargando del primer stream")
	stream = yt.streams.first()
	stream.download()

#proceso para extraer el clip concreto del video
print("[INFO] Extrayendo clip indicado...")
try:
	ffmpeg_extract_subclip("{}.mp4".format(yt.title), args["inicio"], args["final"], targetname="{}.mp4".format(args["name"]))
except:
	ffmpeg_extract_subclip("descarga.mp4", args["inicio"], args["final"], targetname="{}.mp4".format(args["name"]))
#Para listar todos los stream del video
#lstst=yt.streams.all()
#for st in lstst:
#    print(st)

#Finalmente eliminamos el video completo de la carpeta
remove("{}.mp4".format(yt.title))