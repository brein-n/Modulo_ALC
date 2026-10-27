# Corre todos los archivos alc_*_tests.py de esta carpeta y muestra un resumen

import runpy
import sys
import traceback
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
RAIZ = TESTS_DIR.parent

# Para que los tests puedan hacer `from alc import ...`
sys.path.insert(0, str(RAIZ))

archivos = sorted(TESTS_DIR.glob("alc_*_tests.py"))
fallidos = []

for archivo in archivos:
    print(f"\n===== {archivo.name} =====")
    try:
        runpy.run_path(str(archivo), run_name="__main__")
        print(f"OK: {archivo.name}")
    except Exception:
        # flush para que el traceback no salga desordenado respecto de los prints
        sys.stdout.flush()
        traceback.print_exc()
        sys.stderr.flush()
        fallidos.append(archivo.name)

print("\n===== RESUMEN =====")
print(f"Pasaron {len(archivos) - len(fallidos)} de {len(archivos)}")
for nombre in fallidos:
    print(f"FALLÓ: {nombre}")

sys.exit(1 if fallidos else 0)
