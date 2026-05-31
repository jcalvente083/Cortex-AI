import os
import subprocess
import ZZZ_PrepararMemoria as preparar

def ejecutar_bat_en_directorios(directorio_raiz, nombre_bat):
    """
    Recorre un directorio y sus subcarpetas, ejecutando un archivo .bat en cada una.
    """
    # Validar que el directorio base exista
    if not os.path.exists(directorio_raiz):
        print(f"El directorio '{directorio_raiz}' no existe.")
        return

    print(f"Iniciando recorrido en: {directorio_raiz}\n")
    print("-" * 40)

    # os.walk recorre el árbol de directorios de forma recursiva
    for directorio_actual, subdirectorios, archivos in os.walk(directorio_raiz):
        print(f"Procesando carpeta: {directorio_actual}")

        try:
            # subprocess.run ejecuta el comando.
            # - cwd=directorio_actual: Hace que el .bat se ejecute "estando" dentro de esa subcarpeta.
            # - shell=True: Es necesario en Windows para ejecutar archivos .bat correctamente.
            # - capture_output=True y text=True: Capturan lo que el .bat imprime en consola.
            resultado = subprocess.run(
                [nombre_bat], 
                cwd=directorio_actual, 
                shell=True, 
                capture_output=True, 
                text=True
            )
            
            # Comprobamos si la ejecución fue exitosa (código de salida 0)
            if resultado.returncode == 0:
                print(f"  [✓] Éxito.")
                # Si quieres ver lo que imprime el .bat, descomenta la siguiente línea:
                # print(f"      Salida: {resultado.stdout.strip()}")
            else:
                print(f"  [✗] Error. Código de salida: {resultado.returncode}")
                if resultado.stderr:
                    print(f"      Detalle: {resultado.stderr.strip()}")
                
        except Exception as e:
            print(f"  [!] Excepción al intentar ejecutar el .bat: {e}")
        
        print("-" * 40)

# ==========================================
# CONFIGURACIÓN
# ==========================================

# 1. Pon aquí la ruta de la carpeta principal que quieres recorrer
DIRECTORIO_RAIZ = r"C:\Users\jesus\Desktop\UHU\TFG\tfg_def\PROGRAMACION\TFG_JesusDavidCalventeZapata\memoria\MEMORIA" 

# 2. Pon el nombre de tu archivo .bat (como está en el PATH, basta con el nombre)
NOMBRE_ARCHIVO_BAT = "limpiar_latex.bat" 

if __name__ == "__main__":
  
    ejecutar_bat_en_directorios(DIRECTORIO_RAIZ, NOMBRE_ARCHIVO_BAT)
    
    # 6. Tu proceso final
    print("-> Añadiendo portada y generando PDF definitivo...")
    mem = preparar.PrepararMemoria()
    mem.crear_pdf()
    
    print("\nProceso finalizado. ¡PDF generado con éxito!")
    input("Presiona Enter para salir...")
    os.system("cls")