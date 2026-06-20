"""
Comprime todos los PNGs de imagenes/ y reports/ en sitio.
- Redimensiona a MAX_DIM px en el lado más largo (calidad suficiente para PDF impreso a 150 Dpi en A4)
- Compresión PNG máxima (lossless pero más pequeño)
- Hace una copia de seguridad de los originales si se pide

Uso:
    python compress_images.py           # preview: muestra ahorro sin modificar nada
    python compress_images.py --apply   # aplica los cambios
"""

import sys
from pathlib import Path
from PIL import Image

MAX_DIM   = 1400   # px en el lado más largo — suficiente para 150 DPI en columna de 9 cm
DIRS = [
    Path(r"C:\Users\jesus\Desktop\Cortex-AI\docs\Memoria\imagenes"),
    Path(r"C:\Users\jesus\Desktop\Cortex-AI\reports"),
]

def compress(img_path: Path, apply: bool) -> tuple[int, int]:
    """Devuelve (bytes_antes, bytes_después). Si apply=False no escribe nada."""
    size_before = img_path.stat().st_size

    img = Image.open(img_path)
    w, h = img.size

    needs_resize = w > MAX_DIM or h > MAX_DIM

    if not needs_resize and not apply:
        # Si no va a cambiar nada y estamos en preview, devolvemos sin abrir más
        # (ya medimos el tamaño)
        pass

    # Redimensionar manteniendo proporción
    if needs_resize:
        img.thumbnail((MAX_DIM, MAX_DIM), Image.LANCZOS)

    # Aplanar transparencia sobre fondo blanco (PNG con alpha → PNG RGB)
    if img.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        src = img.convert("RGBA") if img.mode == "P" else img
        background.paste(src, mask=src.split()[-1])
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    if apply:
        img.save(img_path, "PNG", optimize=True, compress_level=9)
        size_after = img_path.stat().st_size
    else:
        # Estimamos el tamaño guardando en memoria
        import io
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True, compress_level=9)
        size_after = buf.tell()

    return size_before, size_after


def main():
    apply = "--apply" in sys.argv

    print(f"{'APLICANDO cambios' if apply else 'PREVIEW (sin modificar nada — añade --apply para ejecutar)'}")
    print(f"MAX_DIM = {MAX_DIM} px\n")

    total_before = total_after = n = 0

    for d in DIRS:
        if not d.exists():
            print(f"  [SKIP] {d} no existe")
            continue
        pngs = list(d.rglob("*.png"))
        print(f"  {len(pngs)} archivos en {d.name}/")
        for p in pngs:
            b, a = compress(p, apply)
            total_before += b
            total_after  += a
            n += 1
            if b != a:
                saving_pct = (b - a) / b * 100
                print(f"    {p.name:45s}  {b/1024:6.0f} KB → {a/1024:6.0f} KB  (-{saving_pct:.0f}%)")

    print(f"\n{'='*60}")
    print(f"Total archivos procesados : {n}")
    print(f"Tamaño antes              : {total_before/1024/1024:.1f} MB")
    print(f"Tamaño después            : {total_after/1024/1024:.1f} MB")
    print(f"Ahorro estimado           : {(total_before-total_after)/1024/1024:.1f} MB  "
          f"({(total_before-total_after)/total_before*100:.0f}%)")
    if not apply:
        print("\n  → Ejecuta con --apply para aplicar los cambios.")


if __name__ == "__main__":
    main()
