"""
Programa principal version 0.1
Detectar jugadores y pelota
"""
#importamos los paquetes necesarios
import argparse
import cv2 as cv 
import numpy as np 
import imutils
from imutils.video import FileVideoStream
import time
import math
from math import sqrt
import matplotlib.pyplot as plt 


#construimos los argumentos necesarios para el programa
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True,
	help="ruta del video a analizar")
ap.add_argument("-m", "--modo", default=1, type=int,
	help="modo del programa")
ap.add_argument("-o", "--output", default=1, type=int,
	help="elegir visualización")
ap.add_argument("-j", "--jugador", default=1, type=int,
	help="jugador a realizar estadisticas")
args = vars(ap.parse_args())

def distancia(p1, p2):
	"""
	Función sencilla para calcular la distancia entre puntos
	"""

	dist = sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
	return dist

def line_disc(img,ini, fin, color, grosor):
	"""
	Funcion para dibujar una linea discontinua en una imagen
	"""

	paso_x = abs(ini[0]-fin[0])/16
	paso_y = (fin[1]-ini[1])/16

	for i in range(0,16,4):
		cv.line(img, (int(ini[0]+i*paso_x), int(ini[1]+i*paso_y)), (int(ini[0]+(i+1)*paso_x), int(ini[1]+(i+1)*paso_y)), color, grosor)

	return img

def relaccion_jugadores(pos_ant, pos_act):
	"""
	Función para relacionar un jugador detectado en un frame con 
	el detectado en el frame siguiente. Esta función ayuda al seguimiento
	y a la actualización del filtro de kalman.
	"""

	i = 0

	pos_res = len(pos_ant)*[None]

	for c1 in pos_act:

		j = 0
		cambio = 0
		dist_min = 1000000

		for c2 in pos_ant:
			dist_act = distancia(c1, c2)

			if (dist_act < dist_min and dist_act<50 and pos_res[j]==None):
				dist_min = dist_act
				pos_j = j
				cambio = 1

			j = j + 1

		if (cambio == 1):
			pos_res[pos_j] = c1
		else:
			pos_res.append(c1)

	return pos_res

class FiltroKalman:
	"""
	Clase de filtro de kalman para aplicar a cada uno de los elementos
	a realizar un seguimiento, además de ayudar a la detección en 
	cirscunstancias especiales.
	"""
	def __init__(self):
		q = 10
		r = 1
		self.kf = cv.KalmanFilter(4, 2)
		self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
		self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
		self.kf.measurementNoiseCov = np.array([[r**2, 0],[0, r**2]], np.float32)
		self.kf.processNoiseCov = np.array([[q**2,0,0,0],[0,q**2,0,0],[0,0,q**2,0],[0,0,0,q**2]], np.float32)

	def Estimate(self, coordX, coordY):
		''' Con esta función se estimará la posición del elemento en este instante'''
		measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
		predicted = self.kf.predict()
		if (coordX!=0 or coordY!=0):
			predicted = self.kf.correct(measured)

		return predicted

# FUNCIONES

def condicionRGB(img):
	"""
	Función para aplicar la condición G>R>B para extraer 
	el campo de juego de nuestra imagen
	"""

	b,g,r = cv.split(img)

	#Aplicamos condición g>r>b
	bgExtract1 = g>r
	bgExtract2 = r>b
	bgExtract1 = np.uint8(bgExtract1)
	bgExtract2 = np.uint8(bgExtract2)

	bgExtract = cv.bitwise_and(bgExtract1,bgExtract2) 
	bgExtract = np.uint8(bgExtract)*255

	#Aquí extraemos y guardamos la máscara del campo
	bgMask = np.zeros((height,width), dtype=np.uint8)
	contours, hierarchy = cv.findContours(bgExtract, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
	c = max(contours, key = cv.contourArea)
	cv.drawContours(bgMask, [c], -1, 255, -1)

	bgExtract = cv.bitwise_not(bgExtract)

	return bgExtract, bgMask

def gradiente_sobel(img, umbral):
	"""
	Función donde calcularemos el gradiente de sobel 
	de nuestra imgaen y aplicaremos un umbral 
	"""
	gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
	gray = cv.GaussianBlur(gray,(5,5),0)
	dX = cv.Sobel(gray, cv.CV_32F, 1, 0, (3,3))
	dY = cv.Sobel(gray, cv.CV_32F, 0, 1, (3,3))
	mag, direction = cv.cartToPolar(dX, dY, angleInDegrees=True)
	_, SobelG = cv.threshold(mag,umbral,255,cv.THRESH_BINARY)
	SobelG = SobelG.astype(np.uint8)

	return SobelG

def etiquetado(img, imgOrig):
	"""
	Proceso de etiquetado de cada una de las siluetas extraidas
	en nuestra imagen.
	Eliminamos además aquella mayores de una cierta área.
	"""
	x_ball = 0
	y_ball = 0
	pos_jug = np.array([0,0])
	num_labels, labels_im, stats, centroids = cv.connectedComponentsWithStats(img)

	jugadores = []
	for label in range(num_labels):

		area = stats[label, cv.CC_STAT_AREA]
		alto = stats[label, cv.CC_STAT_HEIGHT]
		ancho = stats[label, cv.CC_STAT_WIDTH]

		densidad = area / (alto*ancho)

		#Caracteristicas de los jugadores
		if (area > 500) and (area < 2000):

			if (alto/ancho > 1 and alto/ancho <3 ):
				if (densidad>0.3 and densidad<0.9):
					x_bb = stats[label, cv.CC_STAT_LEFT]
					y_bb = stats[label, cv.CC_STAT_TOP]
					cv.rectangle(imgOrig, (x_bb, y_bb), (x_bb+ancho, y_bb+alto), (0,0,255), 2)
					cv.circle(imgOrig, (int(centroids[label,0]), int(centroids[label, 1])), 3, (0,255,0), -1)
					jugadores.append([int(centroids[label,0]), int(centroids[label, 1])])

		#Caracteristicas del balon
		if (area > 100) and (area < 300):

			if (alto/ancho > 0.5 and alto/ancho <=1.5 ):
				if (densidad>0.7 and densidad<0.9):
					x_bb = stats[label, cv.CC_STAT_LEFT]
					y_bb = stats[label, cv.CC_STAT_TOP]
					cv.rectangle(imgOrig, (x_bb, y_bb), (x_bb+ancho, y_bb+alto), (0,0,255), 2)
					cv.circle(imgOrig, (int(centroids[label,0]), int(centroids[label, 1])), 3, (255,0,0), -1)
					x_ball = int(centroids[label,0])
					y_ball = int(centroids[label,1])
					#print(area, densidad)
	

	return imgOrig, x_ball, y_ball, jugadores

def TransformadaHough(img, orig, rho, theta, threshold, minLenght, maxGap):
	"""
	Función para calcular la tranformada de hough de nuestra imagen
	a fin de detectar la lineas rectas del terreno de juego y eliminarlas
	"""
	lines = cv.HoughLinesP(img, 
						rho, 
						theta,
						threshold,
						lines = np.array([]),
						minLineLength = minLenght,
						maxLineGap = maxGap)	

	for line in lines:
		for x1, y1, x2, y2 in line:
			#cv.line(img, (x1, y1), (x2, y2), (0,0,0), 8)
			img = line_disc(img, (x1,y1), (x2,y2), (0,0,0), 7)

	return img



#inicializamos el video a leer
print("[INFO] Leyendo clip...")
vs = FileVideoStream(args["video"]).start()
time.sleep(2.0)

jugadores_ant = 0
registro_posiciones=[]

kernal = np.ones((2,2), np.uint8)

KFJug = []
jug_perdido_cont = []
KFBall = FiltroKalman()
predictedCoords = np.zeros((2, 1), np.float32)

if (args["modo"]==1):
	ejecuta = 0
else:
	ejecuta = 1



#Bucle de lectura de cada frame
while True:
	try:
		frame = vs.read()
		height = frame.shape[0]
		width = frame.shape[1]
		
		#Condición g>r>b
		bgExtract, bgMask = condicionRGB(frame)

		#Aplicamos gradiente de sobel
		SobelG = gradiente_sobel(frame, 40)

		#Sumamos resultado de ambos procesosa
		imgFinal = bgExtract + SobelG

		#Aplicamos máscara de región de interés
		imgFinal = cv.bitwise_and(imgFinal,bgMask)

		#Destacamos siluetas con operaciones morfológicas
		imgFinal = cv.morphologyEx(imgFinal,cv.MORPH_CLOSE, kernal, iterations=2)

		#Buscamos rectas predominantes en la imagen
		imgFinal = TransformadaHough(imgFinal, frame, 6, np.pi / 60, 500, 80, 15)

		#Etiquetamos entra las siluetas detectadas, cual correspoende a jugadores y pelota
		frame, x_b, y_b, jugadores_act = etiquetado(imgFinal, frame)

		#Hacemos el seguimiento de los jugadores detectados
		if (jugadores_ant!=0):
			#Relacionamos los jugadores detectados con los detectados anteriormente
			pos_actualizada = relaccion_jugadores(jugadores_ant, jugadores_act)
			print("## NUEVO FRAME ##")
			print(pos_actualizada)

			if (args["modo"]==3):
				#Se guarda la información del jugador seleccionado
				try:
					registro_posiciones.append([int(jugadores_ant[args["jugador"]][0]),int(jugadores_ant[args["jugador"]][1])])
				except:
					print("No existe ese jugador\n")

			if (args["modo"]==1):
				#Ponemos texto identificador encima de cada jugador
				num_jug = 0

				for jugador in pos_actualizada:
					if jugador != None:
						frame = cv.putText(frame, 'Player ' + str(num_jug), (jugador[0], jugador[1]), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv.LINE_AA)
					num_jug = num_jug + 1

			#Agregamos a la lista aquellos jugadores detectados nuevos	
			while (len(KFJug) != len(pos_actualizada)):
				KFJug.append(FiltroKalman())
				jug_perdido_cont.append(0)

			jugadores_ant = pos_actualizada

		else:

			jugadores_ant = jugadores_act
		
		#Estimación con filtro de kalman
		for j in range(len(KFJug)):
			#Comprobamos que el jugador no lleve más de 10 frames sin detectarse
			if jug_perdido_cont[j]<11:
				#Si se ha actualizado la posición, actualizamos Kalman con las nuevas medidas
				if (pos_actualizada[j] != None):
					JugCoords = KFJug[j].Estimate(pos_actualizada[j][0], pos_actualizada[j][1])
					jug_perdido_cont[j] = 0
				#Si no se ha actualizado la posición, realizamos una predicción con Kalman
				else:
					JugCoords = KFJug[j].Estimate(0, 0)
					jugadores_ant[j] = [JugCoords[0], JugCoords[1]]
					jug_perdido_cont[j] = jug_perdido_cont[j]+1

					#Ponemos texto identificado para los jugadores detectados por predicción
					if (args["modo"]==1):
						frame = cv.putText(frame, 'Player ' + str(j), (int(JugCoords[0]), int(JugCoords[1])), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv.LINE_AA)
					cv.rectangle(frame, (int(JugCoords[0])-25, int(JugCoords[1])-50), (int(JugCoords[0])+25, int(JugCoords[1])+50), (0,255,0), 2)

				cv.circle(frame, (int(JugCoords[0]), int(JugCoords[1])), 3, (0,0,255), -1)
			#Damos por perdido el jugador si no aparece durante 10 frames consecutivos
			else:
				jugadores_ant[j] = [0,0]
		
		#Estimación de la pelota con filtro de Kalman
		predictedCoords = KFBall.Estimate(x_b, y_b)
		print(predictedCoords)
		cv.circle(frame, (int(predictedCoords[0]), int(predictedCoords[1])), 3, (0,0,255), -1)
		'''
		print("##Reales##")
		print(x_b, y_b)
		print("##Predichas##")
		print(int(predictedCoords[0]), int(predictedCoords[1]))
		'''

		#Abrimos ventana con la visualización de las distintas imágenes procesadas
		#cv.imshow('Sobel', SobelG)
		#cv.imshow('bgExtract', bgExtract)
		if (args["output"]==1):
			cv.imshow('Original', frame)
		if (args["output"]==2):
			cv.imshow('Discriminacion color', bgExtract)
		if (args["output"]==3):
			cv.imshow('Sobel', SobelG)
		if (args["output"]==4):
			cv.imshow('Procesado Final', imgFinal)

		
		#Comprobamos si qe ha pulsado la tecla para salir
		key = cv.waitKey(ejecuta) & 0xFF 
		if key == ord("q"):
			break
		#Si pulsamos la letra "s" guardaremos el actual frame como una imagen png
		elif key == ord("s"):
			cv.imwrite("Imagenes/last_save.png", frame)

	except:
		"""
		Esta excepción saltará cuando el video llegue a su fin.
		En esta rutina de excepción se finalizarán determinados procesos y se sacarán las estadísticas de la jugada
		"""
		cv.destroyAllWindows()
		vs.stop()

		if (args["modo"]==3):
			mapa_calor = np.zeros((height,width,3), np.uint8)
			mapa_calor[:] = (0,100,0)
			distancia_recorrida = 0
			tiempo_corriendo = 0
			velocidades = []
			for j in range(1,len(registro_posiciones)):
				if (registro_posiciones[j][0]!=0 or registro_posiciones[j][1]!=0):
					cv.line(mapa_calor, (registro_posiciones[j][0],registro_posiciones[j][1]), (registro_posiciones[j-1][0],registro_posiciones[j-1][1]), (0,0,150), 3)
					distancia_actual = distancia((registro_posiciones[j][0],registro_posiciones[j][1]), (registro_posiciones[j-1][0],registro_posiciones[j-1][1]))
					distancia_recorrida = distancia_recorrida + distancia_actual
					velocidades.append(distancia_actual)
					if distancia_recorrida > 3:
						tiempo_corriendo = tiempo_corriendo + 1
				print(distancia_actual)

			print("ESTADÍSTICAS RECOPILADAS DEL VIDEO")
			print("DISTANCIA RECORRIA POR EL JUGADOR:")
			print(distancia_recorrida)
			print("VELOCIDAD MÁXIMA DEL JUGADOR:")
			print(max(velocidades))
			print("TIEMPO QUE HA ESTADO EL JUGADOR CORRIENDO:")
			print(tiempo_corriendo/25)

			plt.plot(range(len(velocidades)),velocidades)
			plt.xlabel('tiempo (1/25 s)')
			plt.ylabel('velocidad (pixel/frame)')
			plt.suptitle('Velocidad del jugador')
			plt.show()
			
			cv.putText(mapa_calor, 'Distancia Total: ' + str(distancia_recorrida), (40,40), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv.LINE_AA)
			cv.imshow('Estadistica 1', mapa_calor)
			cv.waitKey(0)

		break
