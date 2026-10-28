# cordones.py — líneas de cuneta y cordón desde LiDAR aéreo

Infiere la **línea de cuneta** (pie del cordón) y la **línea de cabeza de cordón**
a ambos lados de cada eje de calle, a partir de una nube LiDAR aérea urbana de
densidad media (~20 pts/m²). Genera polilíneas 3D para usar como *breaklines*
en un TIN de Civil 3D (modelación hidráulica 2D).

Con ~22 cm de espaciamiento y 5–10 cm de ruido vertical, un cordón de 15 cm no
se ve punto a punto. El script no busca saltos locales: **apila los puntos de una
ventana a lo largo del eje en un único perfil transversal** y ajusta por
mínimos cuadrados robustos una plantilla calzada–cordón–vereda. Con 4 m de
ventana hay cientos de puntos por lado, y el escalón queda bien determinado.

## Instalación

Requiere Python 3.11 o posterior.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**PDAL** (opcional) solo hace falta si la nube no trae el suelo clasificado
(clase 2). Lo más simple es instalarlo con conda:

```bash
conda install -c conda-forge pdal python-pdal
```

Si los bindings de Python no están, el script prueba con la CLI `pdal`.

## Uso

```bash
# 1) Calibración: un solo eje, con gráfico de control para CADA ventana
python cordones.py --config config.yaml --debug             # eje debug.eje del config
python cordones.py --config config.yaml --debug --eje 2     # por índice
python cordones.py --config config.yaml --debug --eje EJES_2F   # por nombre

# 2) Proceso completo
python cordones.py --config config.yaml

# opciones: --eje X (procesar solo ese eje), --sin-graficos, -v (log detallado)
```

En DXF el nombre de cada eje es `CAPA_HANDLE` (por ejemplo `EJES_2F`). En SHP es el
valor de `campo_nombre`. Los nombres aparecen en el log y en el CSV.

### Probar con datos sintéticos

```bash
python ejemplo/generar_sintetico.py ejemplo/datos     # 2 cuadras, cordón 15/17 cm, ruido σ=7 cm
python cordones.py --config ejemplo/datos/config.yaml --debug
python tests/test_sintetico.py                        # verificación automática contra la verdad
```

En el caso sintético, el error del offset tiene p95 ≈ 5–6 cm y el de la altura
h un RMSE ≈ 1 cm. La bocacalle queda cortada y el rebaje de garaje va a
BAJA_CONFIANZA.

## Datos de entrada

| Dato | Detalle |
|---|---|
| Nube | `.las`/`.laz`, un archivo o un patrón glob de tiles (`tiles/*.laz`). Se usan la clase 2 y, si existen, la intensidad y/o el RGB. |
| Ejes | DXF (LWPOLYLINE/POLYLINE/LINE/SPLINE, los arcos se discretizan) o SHP de líneas. Un eje por calle, aproximadamente centrado. |
| SRC | Ejes y nube en el **mismo** sistema proyectado (p. ej. POSGAR 2007 / faja). No se reproyecta. `epsg` solo se escribe en el LandXML. |

**Izquierda/derecha** se refieren al sentido de digitalización del eje. El
offset es positivo hacia la izquierda.

## Método

1. **Lectura por buffer** (`lectura.py`). Se descartan los tiles cuyo
   encabezado no toca el eje. El resto se lee por bloques y solo se retienen
   los puntos a menos de `ancho/2 + tolerancia + vereda + margen` del eje
   (consulta con `cKDTree`). Con `lectura.cache: true` el recorte se guarda en
   `salida/cache/*.npz`, así la recalibración no relee la nube.
2. **Suelo**. Se usa la clase 2. Con `modo: auto`, si hay menos del 5 % de
   puntos clase 2, se clasifica con PDAL (SMRF). También se puede forzar
   `smrf` o `csf`.
3. **Proyección** (`eje.py`, `apilado.py`). Cada punto pasa a
   (estación, offset, z) respecto del eje, con un índice `cKDTree` sobre
   subsegmentos de 1 m, así que funciona igual en curvas.
4. **Ventanas** de `longitud` m (4 por defecto) cada `paso` m (1 por defecto).
5. **Ajuste por lado** (`ajuste.py`), con u = distancia al eje y ds = estación
   relativa al centro de la ventana:

   ```
   calzada (u < d): z = z0 + g·ds + sc·u
   vereda  (u > d): z = z0 + g·ds + sc·d + h + sv·(u − d)
   ```

   Las incógnitas son `d, h, sc, sv, z0` y además **`g`, la pendiente
   longitudinal dentro de la ventana**. Hace falta porque 1 % de rasante en
   4 m da 4 cm, del orden del cordón, y sin ese término el perfil apilado se
   "ensucia".

   El escalón es una sigmoide de 5 cm de ancho, para que el ajuste sea
   derivable en d. Primero se hace una búsqueda en grilla de d, gruesa y
   luego fina, con los parámetros lineales resueltos por IRLS-Huber. Después
   se refina todo con `least_squares(loss='soft_l1')`, con límites
   `h ∈ [h_min, h_max]` y `d ∈ [ancho/2 ± tolerancia]`.
6. **Modelo nulo y confianza**. Se ajusta el mismo modelo **sin escalón**
   (quiebre de pendiente continuo en d) y se compara por
   `ΔBIC = BIC_nulo − BIC_escalón`, sobre los mismos inliers. El índice de
   confianza (0–1) es el producto de cinco factores:
   - `f_n = min(1, n/n_ref)`, por la cantidad de puntos;
   - `f_rms`: vale 1 si el RMS es menor o igual que `rms_bueno` y 0 si es mayor o igual que `rms_malo`;
   - `f_bic = 1 − exp(−ΔBIC/escala_bic)`, por la significancia del escalón;
   - `f_lim = 0,5` si d o h quedaron pegados a un límite;
   - `f_lados`: el equilibrio de puntos entre calzada y vereda.

   Del jacobiano salen además los errores estándar `se_offset` y `se_h`.
7. **Raster radiométrico** (`raster.py`). Se usa si hay intensidad o RGB. Se
   arma un raster de 0,25 m en el sistema "enderezado" (estación × offset) del
   eje. En cada ventana se toma la mediana de las filas, se busca el máximo
   gradiente asfalto/hormigón dentro del rango de d y su confianza sale de la
   relación gradiente/ruido. La combinación con la estimación geométrica
   funciona así:
   - si las dos estimaciones difieren menos de `tolerancia_combinacion`, se
     promedian ponderadas por confianza;
   - si difieren más, se queda la de mayor confianza y esa confianza se reduce
     a la mitad.

   La cota de cuneta siempre sale de la calzada ajustada, evaluada en el d
   final.
8. **Post-proceso** (`postproceso.py`), por lado:
   - Descarta las ventanas con `conf < conf_min`.
   - Elimina outliers de offset y de z de cuneta con una mediana móvil y un
     umbral.
   - Descarta los tramos aislados de menos de `tramo_min`.
   - Interpola los huecos de menos de `hueco_max` (10 m) y corta la línea en
     los mayores.
   - Suaviza con Savitzky-Golay o con spline.
   - Si h no es confiable (conf baja, `se_h > se_h_max` o h pegada a un
     límite), la interpola si el hueco es corto. Si no, usa `h_tipica` y marca
     la estación como **asumida**.
9. **Reconstrucción**. Para cada estación se calcula `XY = eje(s) ± d·normal`.
   La cuneta toma `Z = z_cuneta` y la cabeza de cordón `Z = z_cuneta + h`.

## Salidas (`salida/`, o `salida/debug/` con `--debug`)

| Archivo | Contenido |
|---|---|
| `cordones.dxf` | POLYLINE 3D en las capas `CUNETA_IZQ`, `CUNETA_DER`, `CORDON_IZQ` y `CORDON_DER`, con las líneas completas: medidas, interpoladas y con h asumida. `BAJA_CONFIANZA` tiene **copias** de los tramos dudosos: offset interpolado, h interpolada o asumida, o conf < `conf_alta`. |
| `cordones_ventanas.csv` | Una fila por ventana y lado: offset geométrico y del raster, offset final, cotas, h, pendientes, rms, ΔBIC, errores estándar y los factores de confianza. Sirve para calibrar. |
| `cordones_lineas.csv` | Los vértices finales: X, Y, Z de cuneta y de cabeza, h, estado (`medido`/`interpolado`/`asumido`) y la marca de dudoso. |
| `cordones.xml` | LandXML 1.2: superficie `CORDONES` con las breaklines en `SourceData`, sin los tramos de BAJA_CONFIANZA porque ya están en las capas principales. |
| `graficos/<eje>/perfil_longitudinal.png` | Offset, h, z de cuneta y confianza a lo largo del eje, en bruto y final. |
| `graficos/<eje>/raster.png` | El raster enderezado con las líneas finales superpuestas. |
| `graficos/<eje>/ventanas/est_*.png` | Perfil apilado sin la pendiente longitudinal, con las medianas cada 10 cm, la plantilla ajustada, el modelo sin escalón, el borde detectado en el raster y el perfil radiométrico. En modo normal se guarda 1 de cada `cada_n_ventanas`; con `--debug`, todas. |

### Llevarlo a Civil 3D

- **DXF**: insertar o referenciar el DXF. En la superficie, *Definition →
  Breaklines → Add* elige las polilíneas 3D de las cuatro capas
  principales. Las de `BAJA_CONFIANZA` son copias para revisar, **no las
  agregues**.
- **LandXML**: *Insert → LandXML* crea una superficie con las breaklines. Se
  pueden copiar a la superficie del terreno.

## Calibración sugerida (modo `--debug` sobre 1–2 cuadras)

1. Revisá algunos `est_*.png`. Las medianas negras tienen que mostrar el
   escalón y la plantilla tiene que seguirlo. Si d cae siempre en un límite,
   ajustá `geometria.ancho_calzada`/`tolerancia_offset` (o usá
   `ejes_parametros` para ese eje).
2. Mirá en `cordones_ventanas.csv` la distribución de `delta_bic`, `rms` y
   `se_h` en tramos buenos y en tramos sin cordón (bocacalles, rebajes).
   Ajustá `escala_bic`, `rms_bueno`/`rms_malo` y `conf_min` para que separen
   bien los dos casos.
3. `ajuste.f_scale` es aproximadamente el ruido vertical (0,05–0,08). Subilo
   si hay mucha vegetación baja o autos mal clasificados.
4. Si hay cuneta de hormigón, el borde de asfalto del raster no coincide con
   el cordón. Comparalo en los gráficos con el d geométrico y corregí
   `raster.correccion_offset` (por ejemplo +0,30 a +0,50), o bajá
   `raster.peso`.
5. Para que el raster solo acepte vereda más clara que el asfalto, usá
   `signo_contraste: vereda_mas_clara`.

## Limitaciones conocidas

- No genera las **curvas de esquina** (ochavas/radios). Las líneas se cortan
  en las bocacalles. Los radios hay que cerrarlos a mano o con *feature lines*.
- **Rebajes y vados**: h queda en el límite inferior y el tramo sale como
  interpolado o asumido en BAJA_CONFIANZA. Conviene revisarlos, porque
  hidráulicamente importan.
- La plantilla supone el cordón como escalón vertical. Una cuneta de hormigón
  con pendiente propia queda absorbida en la calzada. La línea de cuneta es la
  intersección calzada–cara del cordón.
- Con **un único archivo LAS gigante**, se relee el archivo completo una vez por
  eje. Conviene teselarlo antes (por ejemplo, `pdal tile` o `lastile`, tiles de
  250–500 m) y usar un patrón glob. La caché `.npz` evita relecturas al
  recalibrar.
- La clasificación con PDAL (SMRF/CSF) está implementada pero no se probó en
  este entorno. Verificá la clasificación resultante.
- El eje debería estar aproximadamente centrado. Si no lo está, aumentá
  `tolerancia_offset` o definí `ejes_parametros`.

## Estructura

```
cordones.py              CLI y orquestación por eje
config.yaml              parámetros (todos)
cordones_lib/
  config.py              valores por defecto + lectura YAML/JSON
  lectura.py             ejes DXF/SHP, nube por tiles/bloques/buffer, PDAL
  eje.py                 estaciones, normales, proyección (s, o)
  apilado.py             proyección de la nube, ventanas, rasgo radiométrico
  ajuste.py              plantilla, modelo nulo, ΔBIC, confianza
  raster.py              raster enderezado, detección de borde, combinación
  postproceso.py         outliers, huecos, suavizado, estados
  exportacion.py         DXF, CSV, LandXML
  graficos.py            gráficos de control
ejemplo/generar_sintetico.py
tests/test_sintetico.py
```
