# Evaluación del Proyecto

## Nota: 16 / 20

## Evaluación general

El EDA está bien construido y demuestra una comprensión razonable del problema. Se identifican correctamente el target, la unidad de observación, el desbalance de clases y el riesgo de leakage por artista.

Sin embargo, todavía faltan varias validaciones importantes antes de considerar el dataset listo para modelar. La principal debilidad está en la definición exacta del problema y en el tratamiento de `chords`, que es probablemente la variable más importante del proyecto.

## Aspectos positivos

- Buena identificación de `main_genre` como target.
- Correcta detección del riesgo de leakage por artista.
- Buena decisión de realizar el split por `artist_id`.
- Missing values y cobertura del target razonablemente analizados.
- Buena exploración inicial de `chords`.
- El uso de 4-grams permite conservar información secuencial local.
- El notebook está bien organizado y presenta una narrativa clara.

## Correcciones necesarias

### 1. Definir mejor el problema de predicción

`main_genre` es una etiqueta asociada al artista y luego asignada a sus tracks. Deben dejar explícito qué están intentando predecir realmente.

Deben indicar:

- qué representa exactamente `main_genre`;
- si el escenario es predecir el género de una canción de un artista no visto;
- qué información estará disponible al momento de realizar la predicción.

### 2. Profundizar el análisis secuencial de `chords`

La columna `chords` es probablemente la principal fuente de información del problema.

Actualmente utilizan:

- estadísticas agregadas como `n_chords`, `n_unique_chords` o proporciones de tipos de acordes, que eliminan el orden;
- 4-grams, que sí conservan relaciones secuenciales locales.

Esto es un buen comienzo, pero deben responder una pregunta fundamental:

> ¿El orden de los acordes aporta información adicional para distinguir géneros?

Deben explorar al menos:

- bigrams, trigrams o transiciones entre acordes;
- patrones repetidos;
- diferencias entre representaciones que conservan y no conservan el orden;
- qué información aportan las etiquetas estructurales como `<verse>` y `<chorus>`.

No es necesario entrenar todavía una LSTM o Transformer. El objetivo es justificar qué representaciones de `chords` deberían evaluarse posteriormente.

### 3. Revisar la normalización de progresiones armónicas

Actualmente progresiones musicalmente equivalentes pero transpuestas pueden considerarse diferentes.

Por ejemplo:

C - G - Am - F  
D - A - Bm - G

Deben discutir si conviene trabajar con acordes absolutos o con alguna representación relativa que permita reconocer progresiones equivalentes en distintas tonalidades.

No necesariamente deben implementarlo todavía, pero sí identificarlo como una decisión pendiente de modelado.

### 4. Validar el parser de acordes

La función que clasifica los acordes por tipo debe revisarse.

No debería asumirse automáticamente que cualquier token no reconocido corresponde a `major`.

Deben:

- crear una categoría `unknown`;
- medir cuántos tokens no reconoce el parser;
- inspeccionar ejemplos manualmente;
- revisar por qué algunas categorías, como `half-diminished`, tienen frecuencia cero.

Hasta validar esta transformación, las conclusiones sobre proporciones de tipos de acordes deben considerarse preliminares.

### 5. Comparar tracks con y sin target

Una parte importante del dataset no posee `main_genre`.

Deben comparar la población etiquetada y no etiquetada utilizando variables como:

- longitud de la progresión;
- cantidad de acordes únicos;
- década;
- presencia de estructura;
- metadata disponible.

El objetivo es comprobar si el conjunto utilizado para aprendizaje supervisado presenta algún sesgo de selección.

### 6. Mejorar las validaciones de calidad

Además de revisar schema, `NaN` y duplicados, deben inspeccionar:

- valores extremos de `n_chords`;
- tokens de acordes inválidos;
- consistencia de IDs;
- fechas fuera de rango;
- categorías inesperadas;
- progresiones idénticas presentes en artistas diferentes.

En particular, los casos con miles de acordes deben ser investigados y no únicamente ocultados mediante límites en las gráficas.

### 7. Usar lenguaje más riguroso en las conclusiones

Evitar afirmaciones como:

> "Esta feature separa géneros."

o:

> "Esta es una feature importante."

En esta etapa solo se están observando asociaciones exploratorias.

Es preferible utilizar formulaciones como:

> "La variable muestra diferencias entre géneros y es candidata para ser evaluada durante el modelado."

### 8. Añadir un inventario final de features

El EDA debería terminar dejando explícito qué variables se utilizarán, excluirán o requerirán transformación.

| Variable | Acción |
|---|---|
| `chords` | Usar / transformar |
| `artist_id` | Solo para split |
| `main_genre` | Target |
| `genres` | Excluir |
| `rock_genre` | Excluir |
| `n_unique_chords` | Feature candidata |
| n-grams | Feature candidata |
| estructura verso/coro | Investigar |
| proporciones de chord quality | Usar después de validar parser |

## Checklist para la próxima entrega

- [ ] Definir explícitamente el escenario de predicción.
- [ ] Mantener el split por `artist_id`.
- [ ] Profundizar el análisis secuencial de `chords`.
- [ ] Comparar representaciones que conservan y no conservan el orden.
- [ ] Discutir normalización por tonalidad o transposición.
- [ ] Validar completamente el parser de acordes.
- [ ] Comparar tracks etiquetados vs. no etiquetados.
- [ ] Investigar outliers y anomalías semánticas.
- [ ] Revisar progresiones duplicadas entre artistas.
- [ ] Reformular conclusiones demasiado fuertes.
- [ ] Añadir un inventario final de features.

## Comentario final

El EDA tiene una buena base. Para mejorar la nota no necesitan agregar más gráficos, sino cerrar mejor las decisiones metodológicas.

La principal prioridad para la próxima entrega debe ser `chords`: justificar cómo se representará una variable secuencial tan importante, qué información se pierde al utilizar estadísticas agregadas y qué representaciones deberían evaluarse posteriormente durante el modelado.
