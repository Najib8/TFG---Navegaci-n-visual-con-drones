# TFG - Navegación visual con drones

## 1. Instalación

Para instalar este proyecto tienes que llevarte el proyecto a local:
```
git clone https://github.com/Najib8/TFG--Navegacion_visual_con_drones.git

cd TFG--Navegacion_visual_con_drones
```

Para ejecutar los scripts de este proyecto, he utilizado un entorno de conda al que le instalo los paquetes de la siguiente forma:

1. Crear el entorno:
```
conda create -n 'nombre_del_entorno'
```

2. Activar el entorno creado:
```
conda activate 'nombre_del_entorno'
```

3. Instalar `python 3.6.13` en el entorno de conda activo:
```
conda install python=3.6.13
```

4. Instalar los requerimientos definidos en el `requirements.txt` sobre `pip` del entorno de conda activo:
```
pip install -r requirements.txt
```

5. Instalar `cupy 8.3.0` en el entorno de conda activo:
```
conda install cuda=8.3.0
```

6. Instalar `pytorch 1.4.0` en el entorno de conda activo:
```
conda install pytorch=1.4.0
```

Ya tenemos todos los paquetes instalados para poder ejecutar todos los scripts correctamente.


## 2. Preparación de las secuencias de imágenes

Para poder ejecutar los scripts necesitamos tener las secuencias de imágenes preparadas.
Tenemos dos opciones: ejecutar los scripts para evaluar los modelos desarrollados utilizando el conjunto de datos DAVIS2017 o
ejecutar los scripts sobre las imágenes en tiempo real que nos ofrece un dron (al que deberemos estar conectados).
Además los códigos para ejecutar estas dos opciones son levemente diferentes: para DAVIS2017, el código está en la rama `evaluacion-davis` y, para las imágenes del dron, en la rama `stream-dron`.

1. Opción para ejecutar los scripts sobre DAVIS2017:



2. Opción para ejecutar los scripts sobre imágenes en tiempo real del dron: