# Comparaciones ATE–ATT: únicamente clustered

Generador: `Do file/Python/compare_ate_att.py`. La ejecución sin opciones genera las 14 tablas; `--selected-countries` genera únicamente efectos y pesos de los países seleccionados.

| Archivos LaTeX | Contenido |
| --- | --- |
| `table_ate_att_main.tex` | Efectos IRM/APOS, muestra completa, dos columnas por outcome |
| `table_ate_att_main_selected_countries.tex` | Efectos IRM/APOS, muestra restringida a los países seleccionados |
| `table_ate_att_heterogeneity_gates_{outcome}.tex` | Tres tablas GATE: tratamiento, grupo, rango de E. coli, ATE y ATT con errores estándar |
| `table_ate_att_sensitivity_{outcome}.tex` | Tres tablas de sensibilidad: cf_y, cf_d, RV y RV_alpha, en porcentaje, ATE frente a ATT |
| `table_ate_att_convex_weights_{outcome}.tex` | Tres tablas de pesos IRM por componente predictivo y learner |
| `table_ate_att_convex_weights_{outcome}_selected_countries.tex` | Tres tablas de pesos IRM para países seleccionados |

Los países seleccionados son Dominican Republic, Guyana, Honduras y Malawi. La tabla es para su muestra conjunta, no una estimación separada por país.

El script lee resultados de `Output` y `Output/ATT`; no reestima modelos. Empareja por identificadores y exige correspondencia uno a uno. Los pickles comparativos conservan estadísticas y rutas de origen, incluidos intervalos puntuales y conjuntos de GATE. Solo se admiten filas clustered; los pesos se seleccionan por el nombre del modelo IRM clustered. Las tablas usan booktabs y adjustbox.

ATE promedia sobre la población de análisis; ATT sobre observaciones en hogares que usan cualquier tratamiento. En GATE se aplica esa distinción dentro de cada grupo. Los contrastes ATT por método comparten la población objetivo de cualquier tratamiento. Las estrellas no prueban diferencias entre ATE y ATT ni entre grupos. No se realiza inferencia sobre ATT menos ATE.

No hay resultados guardados de GATE o sensibilidad para los países seleccionados, ni pesos APOS, por lo que esas comparaciones no se generan. Los pesos g0, g1 y m se muestran por separado, sin promediarlos entre componentes.

Lectura descriptiva de los resultados actuales: no hay una dirección universal ATE–ATT. En los 12 efectos principales reportados de la muestra completa, 10 diferencias ATT menos ATE son positivas (una de ellas, diarrea IRM, es prácticamente cero); en países seleccionados, 9 de 12 son negativas. Estos conteos son descriptivos y los efectos no son observaciones independientes. En GATE, el ATT de cualquier tratamiento para algún riesgo es más negativo en los cinco grupos observados. La robustez también depende del outcome y del tratamiento. Los pesos IRM son prácticamente iguales entre versiones (diferencia absoluta máxima menor de 0.000001).
