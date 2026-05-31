import os
import glob
import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy.signal import welch
import librosa
from src.config import SR

class ExtraccionCaracteristicas:
    """
    Clase encargada de extraer las características acústicas de los audios. 
    Extraerá características como:
        - JITA          - rJitter       - RAP
        - rPPQ          - rSPPQ         - ShimmerDb
        - Shimmer       - rAPQ          - rSAPQ
        - Hnr           - Nne           - CHNR
        - GNE           - FTRI          - ATRI
        - FFTR          - FATR          -
    
    Además, tendrá métodos para procesar audios individuales o carpetas enteras, y guardar los resultados en CSV.
    """
    def __init__(self):
        pass

    #=================================================================
    # Extracción de Características Acústicas 
    #=================================================================

    def calcular_temblor(self, array_valores, fps):
        """
        Calcula la frecuencia y amplitud del temblor vocal a partir de un array de valores 
        y la frecuencia de muestreo (fps) del array. Se enfoca en el rango típico de temblor vocal (3-15 Hz) y 
        devuelve la frecuencia dominante y un índice de intensidad basado en la amplitud relativa al promedio. 
        Si no hay suficientes datos activos, devuelve None.

        Args: 
            array_valores (np.ndarray): Array de valores del que se extraerá el temblor.
            fps (float): Frecuencia de muestreo del array.
        
        Returns:
            frecuencia_temblor (float): Frecuencia dominante del temblor vocal en Hz.
            indice_intensidad (float): Índice de intensidad del temblor basado en la amplitud relativa al promedio.
        """

        activos = array_valores[array_valores > 0]
        if len(activos) < fps: 
            return None, None
            
        tendencia_removida = activos - np.mean(activos)
        frecuencias, potencias = welch(tendencia_removida, fs=fps, nperseg=min(len(activos), 256))
        
        rango_temblor = (frecuencias >= 3) & (frecuencias <= 15)
        frecuencias_validas = frecuencias[rango_temblor]
        potencias_validas = potencias[rango_temblor]
        
        if len(potencias_validas) == 0:
            return None, None
            
        idx_max = np.argmax(potencias_validas)
        frecuencia_temblor = frecuencias_validas[idx_max] 
        amplitud_temblor = np.sqrt(potencias_validas[idx_max])
        indice_intensidad = (amplitud_temblor / np.mean(activos)) * 100 
            
        return frecuencia_temblor, indice_intensidad

    def extraer_desde_array(self, senal: np.ndarray, sr_origen: int = SR) -> dict:
        """
        Método que recibe un array de audio y su frecuencia de muestreo, y devuelve un diccionario con las características acústicas extraídas.

        Args:
            senal (np.ndarray): Array de audio.
            sr_origen (int): Frecuencia de muestreo del audio.
        Returns:
            dict: Diccionario con las características acústicas extraídas.
        """
        try:
            
            senal = librosa.util.normalize(senal)
            sound = parselmouth.Sound(senal, sampling_frequency=sr_origen)
            
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=600)
            intensity = sound.to_intensity(time_step=0.01)
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
            
            jita = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3) * 1e6
            rJitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
            rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3) * 100
            rppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3) * 100
            rsppq = call(point_process, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3) * 100 

            shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
            rapq = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
            rsapq = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100

            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            nne = -hnr if hnr is not None else None 
            
            cpp = call(sound, "To PowerCepstrogram", 60, 0.002, 5000, 50)
            chnr = call(cpp, "Get CPPS", True, 0.01, 0.001, 60, 330, 0.05, "Parabolic", 0.001, 0.05, "Straight", "Robust")
 
            fps_pitch = 1 / 0.01 
            f0_array = pitch.selected_array['frequency']
            
            
            fftr, ftri = self.calcular_temblor(f0_array, fps_pitch)
            
            amp_array = intensity.values[0, :]
            
            
            fatr, atri = self.calcular_temblor(amp_array, fps_pitch)

            return {
                "JITA": jita, "rJitter": rJitter, "RAP": rap, "rPPQ": rppq, "rSPPQ": rsppq,
                "ShimmerDb": shimmer_db, "Shimmer": shimmer, "rAPQ": rapq, "rSAPQ": rsapq,
                "Hnr": hnr, "Nne": nne, "CHNR": chnr, "GNE": None, 
                "FTRI": ftri, "ATRI": atri, "FFTR": fftr, "FATR": fatr
            }
            
        except Exception as e:
            print(f"Error procesando el array en memoria: {e}")
            return None

    def extraer_desde_archivo(self, audio_path: str) -> dict:
        """
        Método que recibe la ruta de un archivo de audio, lo procesa y devuelve un diccionario con las características acústicas 
        extraídas.

        Args:
            audio_path (str): Ruta del archivo de audio.
        
        Returns:
            dict: Diccionario con las características acústicas extraídas.
        
        """
        
        try:
            print(f"Procesando: {os.path.basename(audio_path)}...")
            
            senal, sr = librosa.load(audio_path, sr=SR)
            
            
            resultados = self.extraer_desde_array(senal, sr)
            
            if resultados:
                
                resultados = {"AudioPath": os.path.basename(audio_path), **resultados}
                
            return resultados
        except Exception as e:
            print(f"Error procesando {audio_path}: {e}")
            return None

    def extraer_caracteristicas_carpeta(self, carpeta_audios: str, ruta_csv_salida: str = "resultados_acusticos.csv"):
        """
        Método que recibe la ruta de una carpeta con archivos de audio, procesa cada uno y guarda los resultados en un CSV.

        Args:
            carpeta_audios (str): Ruta de la carpeta que contiene los archivos de audio.
            ruta_csv_salida (str): Ruta del archivo CSV donde se guardarán los resultados.

        """
        archivos = glob.glob(os.path.join(carpeta_audios, "*.wav"))
        resultados = []
        
        if not archivos:
            print(f"No se encontraron audios en: {carpeta_audios}")
            return
        
        for audio_path in archivos:
            caracteristicas = self.extraer_desde_archivo(audio_path)
            if caracteristicas:
                resultados.append(caracteristicas)
        
        # Conectamos con el guardado final
        if resultados:
            self.guardar_resultados(resultados, ruta_csv_salida)

    def guardar_resultados(self, resultados: list, ruta_salida: str):
        """
        Método que recibe una lista de diccionarios con las características acústicas extraídas y la ruta de salida, y 
        guarda los resultados en un archivo CSV.
        Args:
            resultados (list): Lista de diccionarios con las características acústicas extraídas.
            ruta_salida (str): Ruta del archivo CSV donde se guardarán los resultados.
        """
        df = pd.DataFrame(resultados)
        df.to_csv(ruta_salida, index=False)
        print(f"\n¡Extracción finalizada! Guardado en '{ruta_salida}'. Muestras procesadas: {len(df)}")

def main():
    # 1. Definimos las rutas
    ruta_entrada = "data/processed/NeuroVoz/audios"
    ruta_salida = "data/processed/NeuroVoz/audios_features/caracteristicas_vocales.csv"
    
    # 2. ASEGURAMOS QUE LA CARPETA DE SALIDA EXISTA ANTES DE EMPEZAR
    Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Extraemos
    extractor = ExtraccionCaracteristicas()
    extractor.extraer_caracteristicas_carpeta(ruta_entrada, ruta_salida)

if __name__ == "__main__":
    main()