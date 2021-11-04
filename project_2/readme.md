# Proyecto #2

Crear una pagina Web que use YOLO para la deteccion de situaciones de riesgo entre personas en un videofeed.

- [Proyecto #2](#proyecto-2)
  - [Servidor Flask](#servidor-flask)
  - [Red Neuronal Yolo](#red-neuronal-yolo)
  - [Instalacion](#instalacion)
    - [Descarga de pesos para YOLO](#descarga-de-pesos-para-yolo)
    - [Correr el servidor Flask con la Red YOLO](#correr-el-servidor-flask-con-la-red-yolo)
  - [Creditos](#creditos)

## Servidor Flask

El servidor esta construido usando el framework de Flask para el manejo web de la parte de innovacion.
Esta diseñado para interpretar un stream de frames en lugar de recibir un video mp4 y enviarlo en un sitio web.

## Red Neuronal Yolo

La red neuronal funciona en 2 capas.

Primero usamos la red neuronal YOLO-COCO para la identificacion de personas dentro de un frame de video.
Calculamos con una libreria la distancia euclidiana entre estas 2 personas para determinar si estan siguiendo las normas de sanidad.

En caso de que no tenemos la segunda capa que detecta si esta persona o no esta usando apropiadamente el cubre bocas que relacionamos usando otra version fine-tuned de YOLO-COCO para este proposito que relacionamos con el modelo anterior usando el criterio de IOU (Intersection over Union) para determinar si se trata de la misma persona.

Con base en esto determinamos si el riesgo que representa esta persona es "Alto" o "Bajo".

## Instalacion

Con el proposito de facilitar la instalacion se dejo un **requirements.txt** dentro de la carpeta "app" donde se encuentra el codigo principal del server.

Se recomienda crear un Virtual Environment y tener una version de **cv2** instalada con soporte para **CUDA** para hacer mas amena la experiencia.

Para instalar la mayoria de los requerimientos solo se debe correr con el Venv activado

```
  pip install -r requirements.txt
```

Dejamos un video de instalacion de [CUDA Support para Windows](https://youtu.be/YsmhKar8oOc) por si se precisa.

### Descarga de pesos para YOLO

Como se explico anteriormente es necesario para nuestros 2 modelos los pesos de las NN. Que se pueden descargar de los siguientes links.

  - [Pesos Yolo](https://drive.google.com/file/d/1hOJCzA_UxOU3g8cAMI38KBUO8Ssk6glK/view?usp=sharing)
    - Este debe de ser guardado en el siguiente path **app/videoManager/social_distancing_detector/yolo-coco**
  - [Pesos Yolo FaceMask](https://drive.google.com/file/d/1sxT8bcA6R0FAd5MRBCxeelIXY8JGLwUq/view?usp=sharing)
    - Este debe de ser guardado en el siguiente path **app/videoManager/social_distancing_detector/face_mask**

### Correr el servidor Flask con la Red YOLO

Posteriormente, cuando el ambiente se encuentre listo solamente es necesario correr en el CLI el comando.

```
  python3 app.py
```
### Entrenar la red del Dataset Face Mask

En caso de que se quiera entrenar la red neuronal para la identificación de una persona con o sin cubrebocas se puede ingresar al siguiente Google colab y hacer una copia del archivo

```
  https://colab.research.google.com/drive/1ZIIZwICi3qgxp9aRr-SyYb0DOYSBMC0r?usp=sharing
``` 
## Creditos

Aqui damos creditos de los 2 repositorios de Github que usamos como apoyo para la creacion del servidor Flask y la NN de yolo

 - [MiguelGrinberg](https://github.com/miguelgrinberg/flask-video-streaming)
 - [AibenStunner](https://github.com/aibenStunner/social-distancing-detector)
