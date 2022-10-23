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


## 2. Preparación del código y de las secuencias de imágenes

Para poder ejecutar los scripts necesitamos tener las secuencias de imágenes preparadas.
Tenemos dos opciones: ejecutar los scripts para evaluar los modelos desarrollados utilizando el conjunto de datos DAVIS2017 o
ejecutar los scripts sobre las imágenes en tiempo real que nos ofrece un dron (al que deberemos estar conectados).
Además los códigos para ejecutar estas dos opciones son levemente diferentes: para DAVIS2017, el código está en la rama `evaluacion-davis` y, para las imágenes del dron, en la rama `stream-dron`.

- A) Opción para ejecutar los scripts sobre **DAVIS2017**:

    Primero nos tenemos que traer a local la rama `evaluacion-davis`.
    ```
    git fetch TFG--Navegacion_visual_con_drones
    git branch evaluacion-davis TFG--Navegacion_visual_con_drones/evaluacion-davis
    ```
    
    Una vez tenemos el código correcto en local, nos tenemos que descargar el conjunto de imágenes [DAVIS2017](https://davischallenge.org/davis2017/code.html) y situarlo correctamente en el sistema de archivos:
    ```
    cd MATNet/data
    ln -s ruta_del_DAVIS2017_descargado DAVIS2017
    ```
    
    Por último, sólo queda descargarnos los [modelos con los pesos preentrenados](https://drive.google.com/file/d/1XlenYXgQjoThgRUbffCUEADS6kE4lvV_/view), descomprimirlos y situarlos correctamente en el sistema de archivos:
    ```
    cd ../ckpt/MATNet
    mv ruta_de_los_modelos_descomprimidos/*.pt .
    ```

- B) Opción para ejecutar los scripts sobre imágenes en tiempo real del dron:
    
    