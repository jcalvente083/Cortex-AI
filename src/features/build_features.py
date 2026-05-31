import pandas as pd
import os
from pathlib import Path
# from config import COLUMNAS_ESPERADAS # Descomenta si usas tu config

class IntegradorDatos:
    """
    Clase encargada de integrar los datos acústicos extraídos con los datos clínicos demográficos (Control y Parkinson).
    Proporciona métodos para:  
        1. Integración en memoria para predicciones en tiempo real (API/Servidor).
        2. Integración por lotes para procesamiento de CSVs.
    
    """
    
    def __init__(self):
        '''
        Inicializa el integrador con un mapeo para codificar el sexo y una lista de columnas esperadas.
        '''
        self.mapeo_sexo = {'M': 0, 'F': 1, 'Hombre': 0, 'Mujer': 1}
        

    def _limpiar_y_codificar(self, df: pd.DataFrame, col_sexo: str = "Sex") -> pd.DataFrame:
        """
        Método interno que limpia y codifica el DataFrame resultante, especialmente la columna de sexo.
        
            Args:
            df (pd.DataFrame): DataFrame que contiene los datos a limpiar y codificar.
            col_sexo (str): Nombre de la columna que contiene el sexo, para aplicar el
                            mapeo de codificación.
            
            Returns:
                pd.DataFrame: DataFrame limpio y con la columna de sexo codificada.
        """
        df_limpio = df.copy()
        
        if col_sexo in df_limpio.columns:
            df_limpio[col_sexo] = df_limpio[col_sexo].replace(self.mapeo_sexo)
            
        return df_limpio


    def integrar_en_memoria(self, dict_acustico: dict, dict_clinico: dict) -> pd.DataFrame:
        """
        Método que integra los datos acústicos extraídos con un diccionario de datos clínicos demográficos,
        limpia y codifica el resultado, y devuelve un DataFrame listo para predicciones en tiempo real.
        
            Args:
                dict_acustico (dict): Diccionario con las características acústicas extraídas.
                dict_clinico (dict): Diccionario con los datos clínicos demográficos (Control
                                o Parkinson) correspondientes al audio procesado.
            
            Returns:
                pd.DataFrame: DataFrame con los datos integrados, limpio y codificado, listo para predicciones.

        """
        datos_completos = {**dict_acustico, **dict_clinico}
        df_paciente = pd.DataFrame([datos_completos])
        df_paciente = self._limpiar_y_codificar(df_paciente, col_sexo="Sex")
        
        return df_paciente


    def integrar_csvs(self, ruta_acustico: str, ruta_hc: str, ruta_pd: str, ruta_salida: str, 
                      col_audio="Audio", col_edad="Age", col_sexo="Sex"):
        """
        Método que integra los datos acústicos extraídos con los datos clínicos demográficos de los CSVs de Control y Parkinson,
        limpia y codifica el resultado, y guarda un nuevo CSV listo para análisis o modelado.

        Args:
            ruta_acustico (str): Ruta del CSV que contiene las características acústicas extraídas.
            ruta_hc (str): Ruta del CSV que contiene los datos clínicos demográficos de los controles sanos.
            ruta_pd (str): Ruta del CSV que contiene los datos clínicos demográficos de los pacientes con Parkinson.
            ruta_salida (str): Ruta del archivo CSV donde se guardarán los resultados integrados.
            col_audio (str): Nombre de la columna que contiene el nombre del archivo de audio en los CSVs clínicos.
            col_edad (str): Nombre de la columna que contiene la edad en los CSVs clínicos.
            col_sexo (str): Nombre de la columna que contiene el sexo en los CSVs clínicos.

            Returns:
                pd.DataFrame: DataFrame con los datos integrados, limpio y codificado, listo para análisis o modelado.

        """
        

        try:
            # 1. Cargar datos
            df_acustico = pd.read_csv(ruta_acustico)
            
            df_hc = pd.read_csv(ruta_hc)
            df_hc['Target'] = 0  # Control
            
            df_pd = pd.read_csv(ruta_pd)
            df_pd['Target'] = 1  # Parkinson
            
            # 2. Unir los clínicos
            df_clinico = pd.concat([df_hc, df_pd], ignore_index=True)
            
            # 3. Limpieza de nombres
            def limpiar_nombre(ruta):
                nombre = os.path.basename(str(ruta).strip()) 
                return nombre if nombre.lower().endswith('.wav') else f"{nombre}.wav"
                
            df_clinico[col_audio] = df_clinico[col_audio].apply(limpiar_nombre)
            
            cols_deseadas = [col_audio, col_edad, col_sexo, 'Target'] 

            cols_existentes = [col for col in cols_deseadas if col in df_clinico.columns]
            
            df_clinico = df_clinico[cols_existentes]
            

            df_final = pd.merge(
                df_acustico, 
                df_clinico, 
                left_on="AudioPath", 
                right_on=col_audio, 
                how="left" 
            )
            
            df_final = df_final.drop(columns=[col_audio])

            df_final = self._limpiar_y_codificar(df_final, col_sexo=col_sexo)

            # ========================================================
            # --- MODIFICACIÓN: EXTRACCIÓN DEL ID_PACIENTE AQUÍ ---
            # ========================================================
            df_final['ID_Paciente'] = df_final['AudioPath'].apply(lambda x: str(x).split('_')[0] + "_" + str(x).split('_')[2].split('.')[0])

            # He añadido 'ID_Paciente' a esta lista para que quede en las primeras columnas del CSV
            cols_frente = ['AudioPath', 'ID_Paciente', 'Target', col_edad, col_sexo] 

            cols_resto = [col for col in df_final.columns if col not in cols_frente]

            df_final = df_final[cols_frente + cols_resto]

            Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
            df_final.to_csv(ruta_salida, index=False)
            
            print(f"¡Éxito! Archivo guardado como '{ruta_salida}'")

            sin_datos = df_final[col_edad].isna().sum()
            if sin_datos > 0:
                print(f"--- Aviso --- \nHay {sin_datos} audios con datos vacíos por no existir en los CSVs demográficos.")
            
            return df_final
            
        except Exception as e:
            print(f"Error durante el proceso: {e}")
            return None
        
def main():
    integrador = IntegradorDatos()
    df_listo = integrador.integrar_csvs(
        ruta_acustico="data/processed/NeuroVoz/audios_features/caracteristicas_vocales.csv",
        ruta_hc="data/raw/NeuroVoz/metadata/data_hc.csv",
        ruta_pd="data/raw/NeuroVoz/metadata/data_pd.csv",
        ruta_salida="data/processed/NeuroVoz/audios_features/datasetFinal.csv"
    )

if __name__ == "__main__":
    main()