import os

try:
    from pypdf import PdfWriter
except ModuleNotFoundError:
    from PyPDF2 import PdfWriter

class PrepararMemoria():

    def __init__(self, WORD_PORTADA = "TFG_PORTADA.docx",PDF_PORTADA = "TFG_PORTADA.pdf", PDF_LATEX = "main.pdf",PDF_FINAL = "TFG.pdf"):
        self.WORD_PORTADA =  WORD_PORTADA
        self.PDF_PORTADA = PDF_PORTADA
        self.PDF_LATEX = PDF_LATEX
        self.PDF_FINAL = PDF_FINAL
    

    def compilar_tfg(self, ruta_word, ruta_portada_pdf, ruta_main, ruta_salida):
        print("--- INICIANDO FUSIÓN DEL TFG ---")
        
        # 1. Comprobar y gestionar la portada
        if os.path.exists(ruta_portada_pdf):
            print(f"[OK] Portada PDF detectada: '{ruta_portada_pdf}'. Se usará esta.")
        else:
            print(f"[!] No se encontró '{ruta_portada_pdf}'. Buscando el documento Word...")
            
            if os.path.exists(ruta_word):
                print(f"[*] Convirtiendo '{ruta_word}' a PDF. Por favor, espera...")
                try:
                    from docx2pdf import convert
                    # Esto abrirá Word en segundo plano un instante para hacer la conversión
                    convert(ruta_word, ruta_portada_pdf)
                    print("[OK] ¡Conversión de la portada completada!")
                except Exception as e:
                    print(f"[ERROR] Falló la conversión de Word a PDF. Detalle: {e}")
                    return
            else:
                print(f"[ERROR] No se encuentra ni '{ruta_portada_pdf}' ni '{ruta_word}'.")
                return

        # 2. Comprobar el documento LaTeX
        if not os.path.exists(ruta_main):
            print(f"[ERROR] No se encuentra el PDF principal de LaTeX: '{ruta_main}'. ¡Compila tu LaTeX primero!")
            return
        else:
            print(f"[OK] Documento principal detectado: '{ruta_main}'.")

        # 3. Unir los documentos
        print("[*] Uniendo la portada y el documento principal...")
        merger = PdfWriter()

        try:
            # El orden aquí es vital: primero añadimos la portada, luego el contenido
            merger.append(ruta_portada_pdf)
            merger.append(ruta_main)

            # Escribimos el archivo final
            with open(ruta_salida, "wb") as archivo_salida:
                merger.write(archivo_salida)

            print(f"[ÉXITO] ¡TFG listo! Guardado como: '{ruta_salida}'")
            
        except Exception as e:
            print(f"[ERROR] Ocurrió un problema al unir los PDFs: {e}")
        finally:
            merger.close()

    def crear_pdf(self):
        self.compilar_tfg(self.WORD_PORTADA, self.PDF_PORTADA, self.PDF_LATEX, self.PDF_FINAL)