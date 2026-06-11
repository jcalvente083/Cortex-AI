import librosa
import soundfile as sf
import numpy as np
import noisereduce as nr
from pathlib import Path
import io
from src.config import SR, TOP_DB, MONO


class PreprocesadorAudio:
    """
    Clase encargada de preprocesar señales de audio para su posterior análisis o inferencia.
    El preprocesamiento incluye:
        1. Conversión a Mono (si es necesario) utilizando librosa.to_mono.
        2. Reducción de ruido utilizando la biblioteca noisereduce.
        3. Recorte de silencios al inicio y al final de la señal utilizando librosa.effects.trim.
    """


    # ==========================================
    # 1. PROCESAMIENTO DE SEÑAL EN MEMORIA
    # ==========================================
    def preprocesarAudio(self, senal: np.ndarray, sr_origen: int) -> np.ndarray:
        """
        Preprocesa una señal de audio en memoria, sin necesidad de escribirla en disco.
        Realiza tareas de conversión a Mono, Resampling a SR, Reducción de ruido y Recorte de silencios 
        
        Args:
            senal (np.ndarray): Señal de audio como un array de numpy.
            sr_origen (int): Frecuencia de muestreo original de la señal.
        
        Returns:
            np.ndarray: Señal de audio preprocesada, lista para ser utilizada en el modelo.
        """
        # Convertir a mono si es necesario
        if MONO and senal.ndim > 1:
            senal = librosa.to_mono(senal)

        # Resamplear si la frecuencia de muestreo no coincide con la esperada
        if sr_origen != SR:
            senal = librosa.resample(senal, orig_sr=sr_origen, target_sr=SR)

        # Reducción de ruido 
        senal_reducida = nr.reduce_noise(y=senal, sr=SR, stationary=True)
        
        # Recortes de silencio al inicio y al final
        senal_recortada, _ = librosa.effects.trim(senal_reducida, top_db=TOP_DB)


        return senal_recortada

    # ==========================================
    # 2. PROCESAMIENTO DE ARCHIVOS EN DISCO
    # ==========================================
    def procesar_archivos(self, ruta_origen, ruta_destino=None):
        """
        Procesa un archivo de audio desde una ruta de origen, aplicando el preprocesamiento definido en la función preprocesarAudio.
        Si se proporciona una ruta de destino, el archivo preprocesado se guardará en esa ubicación.
        Si no se proporciona una ruta de destino, la función simplemente devolverá la señal preprocesada sin guardarla en disco.

        Args:
            ruta_origen (str): Ruta del archivo de audio original que se desea procesar.
            ruta_destino (str, opcional): Ruta donde se guardará el archivo de audio preprocesado. Si no se proporciona, no se guardará el archivo. 

        Returns:
            np.ndarray: Señal de audio preprocesada, lista para ser utilizada en el modelo. 
                        Si ocurre un error durante el procesamiento, se devuelve None.

        """
        try:
            senal, sr_origen = librosa.load(ruta_origen, sr=SR, mono=MONO)
 
            senal_limpia = self.preprocesarAudio(senal, SR)

            if ruta_destino:
                Path(ruta_destino).parent.mkdir(parents=True, exist_ok=True)
                sf.write(ruta_destino, senal_limpia, SR)
                
            return senal_limpia
            
        except Exception as e:
            print(f" --- Error --- \n procesando {ruta_origen}: {e}")
            return None


    # ==========================================
    # 3. PROCESAMIENTO DE DIRECTORIOS COMPLETOS
    # ==========================================
    def procesar_directorio_completo(self, dir_origen, dir_destino):
        """Procesa todos los archivos de audio en un directorio de origen, aplicando el preprocesamiento definido en la función preprocesarAudio.
        Los archivos preprocesados se guardarán en el directorio de destino, manteniendo la misma estructura de archivos.
        
        Args:
            dir_origen (str): Ruta del directorio que contiene los archivos de audio originales.
            dir_destino (str): Ruta del directorio donde se guardarán los archivos de audio preprocesados.
            
        """
       
        ruta_origen = Path(dir_origen)
        ruta_destino = Path(dir_destino)

        for archivo in ruta_origen.rglob('*.wav'):
            ruta_relativa = archivo.relative_to(ruta_origen)
            destino_completo = ruta_destino / ruta_relativa

            if not destino_completo.exists():
                self.procesar_archivos(archivo, destino_completo)

    def procesar_varios_directorios(self, lista_dir_origen, lista_dir_destino):
        """
        Procesa varios directorios de audio, aplicando el preprocesamiento definido en la función preprocesarAudio.
        Los archivos preprocesados se guardarán en los directorios de destino correspondientes, manteniendo la misma estructura de archivos.
        
        Args:
            lista_dir_origen (list): Lista de rutas de los directorios que contienen los archivos de audio originales.
            lista_dir_destino (list): Lista de rutas de los directorios donde se guardarán los archivos de audio preprocesados. 
                                      Debe tener la misma longitud que lista_dir_origen.
            
        """
        for dir_origen, dir_destino in zip(lista_dir_origen, lista_dir_destino):
            self.procesar_directorio_completo(dir_origen, dir_destino)


def main():
    print("Iniciando preprocesamiento de audio...")
    preprocesador = PreprocesadorAudio()

    datasets = [
        ("data/raw/NeuroVoz", "data/interim/NeuroVoz"),
        ("data/raw/PC-GITA",  "data/interim/PC-GITA"),
    ]

    for origen, destino in datasets:
        print(f"\nProcesando: {origen}")
        preprocesador.procesar_directorio_completo(origen, destino)

if __name__ == "__main__":
    main()