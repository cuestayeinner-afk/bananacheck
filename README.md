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