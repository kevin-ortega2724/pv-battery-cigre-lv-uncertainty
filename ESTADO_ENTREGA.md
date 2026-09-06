# Entrega del proyecto para Git

Este paquete es una instantánea del proyecto, no una certificación de resultados
ni una versión lista para envío editorial. Este documento prevalece sobre las
afirmaciones de finalización presentes en borradores anteriores del manuscrito.

## Contenido

Código, configuraciones, pruebas, resultados, datos derivados, fuentes LaTeX,
figuras y PDFs existentes. El empaquetador incluye también el archivo de datos
original suministrado cuando está disponible, en data/raw/. No se incluyen
.git, .venv, .tools, cachés ni temporales. Las dependencias se reinstalan desde
requirements.txt. Los PDFs existentes son anteriores a las últimas modificaciones
LaTeX y no representan necesariamente el texto fuente actual.

## Estado científico que debe corregirse antes de publicar

- S0--S4 y validación de red: se conservan los resultados previamente existentes.
- NSGA-II: existen 30 ejecuciones de cribado y cromosomas guardados. El evaluador
  recorta el banco a dos días y toma muestras cada 24 intervalos (6 horas), usando
  un paso energético de 0.25 h. Ese desacoplamiento y la indexación del despacho
  impiden interpretar sus objetivos como métricas diarias completas.
- El archivo nsga_pareto_front.csv incluye poblaciones finales inviables cuando
  pymoo no devuelve soluciones factibles; no todas sus filas son un frente Pareto.
- El modelo de envejecimiento usa coeficientes sin calibración documentada. No
  constituye una implementación validada del modelo de Schmalstieg.
- paired_policy_statistics.csv usa constantes de referencia, incluidas cifras
  de costo, tensión y degradación sin respaldo en replay diario. Sus p-valores
  NO constituyen evidencia publicable. El ajuste implementado tampoco es Holm
  secuencial y el tamaño de efecto pierde el signo.
- nsga_statistics.csv compara contra una mediana interna. No compara S2--S4.
- scenario_convergence.csv resume identificadores de días, no convergencia de
  objetivos. La duración de almacenamiento no se aplica en el evaluador.
- weighted_sum_comparator.csv selecciona candidatos NSGA-II por suma ponderada;
  no proviene de un optimizador independiente.
- tariff_sensitivity.csv es un reescalado del costo a despacho fijo; no se
  reoptimizó la política bajo tarifas distintas.
- Las pruebas existentes verifican componentes y presencia de archivos; sus
  resultados no validan las conclusiones científicas anteriores.

Pendiente: corregir el evaluador, validar envejecimiento, ejecutar las políticas
sobre días comunes completos de 15 minutos, generar inferencia pareada válida,
completar convergencias/sensibilidades y corregir las afirmaciones prematuras del
manuscrito. También faltan datos administrativos de autores y recompilar PDFs.

## Uso con Git

Extraer el ZIP y abrir la carpeta SISTEMAS_DE_POTENCIA. Revisar este documento
y README.md; ejecutar git init si se desea un repositorio nuevo. Revisar git
status antes de añadir y confirmar archivos. No se ha creado ni publicado ningún
repositorio remoto. El .gitignore existente excluye datos crudos/procesados y
dependencias; esos datos permanecen disponibles localmente al extraer el ZIP.

MANIFEST_SHA256.json permite verificar cada archivo incluido. Los ZIP editoriales
anteriores se incluyen como archivos históricos, no como entregas actualizadas.
