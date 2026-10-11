# BananaCheck

Aplicación local para analizar fruta y terreno, guardar los resultados, comparar indicadores y administrar fincas.

## Requisitos

- Python 3.10 o posterior.
- En Debian/Ubuntu, el paquete de venv de la versión instalada de Python (por ejemplo `python3.12-venv`) para crear `.venv`.
- Ollama y el modelo configurado en la interfaz para el análisis conversacional.
- Los modelos YOLO son opcionales; consulta `SETUP_YOLO.md`.

## Instalación y ejecución

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python bananacheck_server.py
```

Abre `http://127.0.0.1:9090`. En la primera visita configura la cuenta administradora. En Linux/macOS se puede definir una clave staff privada antes de arrancar:

```bash
BANANACHECK_STAFF_PASSWORD='usa-una-clave-larga-y-unica' .venv/bin/python bananacheck_server.py
```

No expongas el servidor a Internet con la clave predeterminada `STAFF`. Para acceso público utiliza HTTPS con un proxy/túnel mantenido y cambia la clave antes de abrir el acceso.

## Datos y migración

El servidor crea `.bananacheck_data/bananacheck.db` fuera de las rutas públicas. En el primer arranque importa los registros existentes de CSV y las cuentas de `.bananacheck_cuenta.json` una sola vez. La migración no elimina ni sobrescribe los archivos antiguos; quedan como copia de origen. La base local, los logs y las copias están ignorados por Git y sus URLs se bloquean explícitamente.

## Informes y exportación

- Después de guardar un análisis de fruta o terreno, las cuentas con rol `USUARIO` pueden descargar el PDF de su ficha. El informe incluye resultados y una gráfica del porcentaje estimado de salud.
- En “Mi historial”, `USUARIO` puede exportar sus propios análisis a CSV y filtrar por tipo, fechas y finca.
- `ADMIN` y `STAFF` no tienen acceso a la descarga PDF ni a la exportación CSV. Sus métricas y funciones de administración permanecen en el panel staff.

El porcentaje de salud es orientativo y se calcula con indicadores visibles; no reemplaza una evaluación agronómica.

## Preclasificación de exportación

La salida visual puede distinguir **Internacional**, **Nacional**, **Rechazo** y **No concluyente** cuando hay etiquetas de mercado revisadas. Internacional implica Nacional; Nacional no implica Internacional. Mientras solo se disponga de las etiquetas generales **Exportacion/Rechazo**, una predicción “Exportacion” se mostrará como apta en ambos cuadros según el criterio general del dataset. Esto mantiene coherentes el diagnóstico y las tarjetas, pero no representa una evaluación independiente de cada mercado. La foto tampoco verifica inocuidad, residuos, trazabilidad, calibre real ni requisitos no visibles.

Para distinguir mercados, prepara imágenes revisadas en estas carpetas:

```text
dataset_exportacion_clasificado/
  Internacional/
  Nacional/
  Rechazo/
```

Etiqueta cada imagen con criterios escritos y revisión humana. Conserva los originales y copia únicamente imágenes revisadas a las carpetas nuevas. Usa nombres `IMG_YYYYMMDD...` para mantener juntas las fotos de una fecha; hacen falta al menos dos sesiones por categoría. Si tienes el lote o racimo, usa un prefijo compartido como `LOTE-ABC_` o `RACIMO-ABC_` en todos sus archivos para que no se separen aunque tengan fechas distintas.

Si esas carpetas aún no existen, el script usa como **preclasificador general** las etiquetas actuales `dataset/Exportacion/` y `dataset/Rechazo/`. Según `dataset/Referencia.txt`, indican bananos considerados aptos o no aptos para exportar. Una predicción general positiva se refleja como “Sí” en ambos cuadros por la regla acordada; para aprender cuándo una fruta es solo nacional, añade imágenes revisadas en las carpetas nuevas. Las imágenes actuales también están separadas por fecha según su clase (Exportacion: 15 y 18 de noviembre de 2021; Rechazo: del 11 al 14 de noviembre de 2021), lo que puede sesgar el modelo por condiciones de captura. Hay que reunir ejemplos de ambas clases en las mismas sesiones antes de confiar en su generalización.

El archivo `modelo_exportacion.pt` generado en esta copia es un checkpoint experimental de **una época**, creado para comprobar el flujo de inferencia; no debe usarse para decisiones de calidad. Aunque esa época obtuvo 99,3% de exactitud en la partición automática, la separación por fecha coincide con las clases y puede inflar artificialmente el resultado. Completa el entrenamiento y evalúa con lotes/fechas representativos e independientes antes de confiar en el modelo.

Con `requirements_yolo.txt` instalado, prepara y entrena el modelo:

```bash
python entrenar_exportacion.py
```

El script crea `dataset_clasificacion/exportacion_binaria/` para el preclasificador actual o `dataset_clasificacion/tres_categorias/` para etiquetas revisadas. Usa una semilla fija y separa sesiones completas. No sobrescribe una carpeta de preparación sin el marcador generado por el script. El modelo queda en `modelo_exportacion.pt` en la raíz del proyecto y el servidor lo carga automáticamente.

El sistema se abstiene por debajo de 0,70 de confianza y muestra **No concluyente**. Ese valor es un umbral inicial, no una probabilidad calibrada ni una garantía de calidad; revísalo con imágenes de validación nuevas y casos difíciles antes de usar la salida. Para mejorar las fotos, captura el fruto completo, enfocado, con luz uniforme y sin filtros; si es posible, fotografía dos caras. La decisión sigue siendo orientativa y requiere inspección humana.

## Métricas del modelo

El panel staff lee `results.csv` y muestra el máximo registrado por métrica y su época. Cada máximo puede proceder de una época diferente; no es un único checkpoint ni una garantía sobre una predicción individual.

En el `results.csv` incluido, los máximos observados son:

| Métrica | Mejor valor | Época |
| --- | ---: | ---: |
| Precisión de detección | 0,87% | 25 |
| Recall de detección | 100% | 1 |
| mAP50 de detección | 49,04% | 33 |
| mAP50-95 de detección | 30,34% | 12 |
| Precisión de segmentación | 0,87% | 25 |
| Recall de segmentación | 100% | 1 |
| mAP50 de segmentación | 49,07% | 33 |
| mAP50-95 de segmentación | 37,71% | 33 |

La precisión observada es baja; valida el dataset y el modelo antes de usar sus predicciones para decisiones de calidad.

## Respaldos

Crear un snapshot consistente y comprobarlo:

```bash
.venv/bin/python backup_datos.py
.venv/bin/python backup_datos.py --verify .bananacheck_data/backups/bananacheck-AAAAMMDD-HHMMSS.db
```

Para restaurar, detén el servidor, conserva una copia del `bananacheck.db` actual, reemplázalo por el respaldo verificado y vuelve a iniciar el servicio. No sincronices las bases con contraseñas ni registros personales a Git.

## Pruebas

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Las pruebas verifican migración SQLite, persistencia y filtros, login STAFF/usuario, limitación de intentos, aislamiento de exportaciones por usuario, denegación para ADMIN/STAFF y descarga PDF.

## Seguridad y operación

- Los intentos fallidos de login se limitan a cinco por IP en una ventana de cinco minutos.
- Las peticiones JSON se limitan a 1 MiB y las imágenes a 10 MiB.
- La actividad se escribe en `.bananacheck_data/bananacheck.log` con rotación; nunca se registran contraseñas.
- No publiques credenciales ni bases de datos. Para operación pública, coloca el servidor detrás de HTTPS y añade controles de red/actualización adecuados.