# Algoritmo Genético para Optimización de Polinomios

Este proyecto implementa un **algoritmo genético (AG)** que busca el valor de `x` que optimiza (minimiza o maximiza) un polinomio paramétrico:

\[
F(x) = a_0 + a_1x + a_2x^2 + \dots + a_nx^n
\]

El programa está escrito en **Python 3** y utiliza una representación binaria de los cromosomas, selección por ruleta, cruza de un punto y mutación por inversión de bits. Además genera un gráfico de convergencia que muestra la mejor evaluación de cada generación.

## Requisitos

- Python 3.8 o superior
- `matplotlib` para generar el gráfico de convergencia

Instalar dependencias (opcional para generar gráficos):

```bash
pip install -r requirements.txt
```

## Uso

Ejecutar el programa principal y seguir las instrucciones en pantalla:

```bash
python main.py
```

Se solicitarán los siguientes datos de entrada:

1. **Grado del polinomio**.
2. **Coeficientes** `a0 .. an` separados por espacios.
3. **Bandera de optimización** (`0` = minimizar, `1` = maximizar).
4. **Intervalo de búsqueda** `[b, c]`.
5. **Número de bits por cromosoma** (precisión de la representación binaria).
6. **Tamaño de la población**.
7. **Número máximo de generaciones**.

El programa mostrará:

- La población final (valores de `x`).
- El mejor cromosoma encontrado junto con su `x` y `F(x)` asociado.
- Un archivo `convergence.png` con el gráfico de la evolución del mejor valor de `F(x)` por generación.

## Ejemplo

Para optimizar el polinomio \(6x^3 - x^2 + x\) en el intervalo `[0, 10]` buscando el **máximo**:

```
Grado del polinomio: 3
Coeficientes a0 .. a3 separados por espacios: 0 1 -1 6
Bandera (0 = minimizar, 1 = maximizar): 1
Extremo inferior del intervalo de búsqueda (b): 0
Extremo superior del intervalo de búsqueda (c): 10
Número de bits por cromosoma: 32
Tamaño de la población: 50
Número máximo de generaciones: 100
```

## Polinomios de prueba sugeridos

- \(6x^3 - x^2 + x\) en `[0, 10]` (maximizar).
- \(-2x^2 + x + 2\) en `[-1, 1]` (maximizar).
- \(-0.25x^4 + 8x^3 + 150x^2 + 50x - 25.6\) en `[0, 6]` (minimizar).
- \(-0.125x^5 + 4x^4 + 50x^2 + 80x + 135\) en `[-10, 10]` (maximizar).

## Estructura del proyecto

```
Algoritmo_MonoObjetivo/
├── main.py           # Implementación del algoritmo genético
├── hello.py          # Archivo de ejemplo (no usado por el algoritmo)
├── requirements.txt  # Dependencias del proyecto
└── README.md         # Este documento
```

## Notas

- Si `matplotlib` no está instalado, el algoritmo se ejecutará igualmente pero no generará el gráfico de convergencia.
- El gráfico `convergence.png` se sobrescribirá cada vez que se ejecute el programa.
- Los parámetros del AG (probabilidad de cruza y mutación) están definidos dentro de `main.py` y pueden ajustarse según necesidad.

