# ==============================================
# VERSIÓN GUIADA PARA ESTUDIO E IMPRESIÓN
# ==============================================
# Este archivo es el MISMO programa original, pero con comentarios
# muy detallados en español.
# IMPORTANTE: no se cambió NINGUNA línea de código funcional; solo
# se agregaron comentarios (líneas que comienzan con #) para explicar
# qué hace cada parte paso a paso.

"""Punto de entrada principal para la optimización de polinomios mediante un algoritmo genético."""

# ------ LIBRERÍAS Y TIPOS BÁSICOS ------
# 'random' nos permite generar valores aleatorios (útiles para el algoritmo genético).
import random
# 'typing' aporta anotaciones de tipo (List, Tuple) para leer mejor qué espera cada función.
from typing import List, Tuple

# 'numpy' es una librería matemática. Aquí la usamos para derivar el polinomio y buscar extremos.
import numpy as np

# La librería 'matplotlib' se usa opcionalmente para generar un gráfico de convergencia.
# Si no está instalada, el programa sigue funcionando sin el gráfico (esto ayuda en PCs donde
# no se puede instalar todo).
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - solo se ejecuta si falta la librería
    # Si hay cualquier problema importando matplotlib, simplemente marcamos 'plt' como None
    # y más adelante evitamos intentar graficar.
    plt = None


# ------ FUNCIONES AUXILIARES SOBRE EL POLINOMIO ------

def polynomial_value(x: float, coefficients: List[float]) -> float:
    """Devuelve F(x) para un polinomio definido por sus coeficientes.

    coefficients[i] corresponde al coeficiente a_i de x**i.
    El polinomio se evalúa utilizando una suma simple.
    """
    # La idea: si el polinomio es a0 + a1*x + a2*x^2 + ...
    # 'enumerate(coefficients)' nos da (i, coef) para i = 0..n
    # y sumamos coef * (x ** i) para armar el valor total F(x).
    return sum(coef * (x ** i) for i, coef in enumerate(coefficients))


def decode(bitstring: str, lower: float, upper: float, bits: int) -> float:
    """Traduce un cromosoma binario a un valor real de x dentro de [lower, upper]."""
    # Un "cromosoma" aquí es una cadena de '0' y '1' de longitud 'bits'.
    # Lo tomamos como un número binario y lo convertimos a entero.
    max_int = 2 ** bits - 1          # mayor entero representable con 'bits' (por ej., con 3 bits es 7)
    integer_value = int(bitstring, 2) # convierte la cadena binaria a entero base 2
    # Luego escalamos ese entero al intervalo real [lower, upper].
    # Si integer_value == 0, queda en 'lower'; si == max_int, llega a 'upper'.
    return lower + (integer_value / max_int) * (upper - lower)


def encode(x: float, lower: float, upper: float, bits: int) -> str:
    """Convierte un valor real x en su representación binaria de cromosoma."""
    max_int = 2 ** bits - 1
    # Normalizamos x al rango [0,1] y lo llevamos a [0, max_int].
    scaled = int(round((x - lower) / (upper - lower) * max_int))
    # Lo formateamos como cadena binaria de longitud fija 'bits', completando con ceros a la izquierda.
    return format(scaled, f"0{bits}b")


def initial_population(pop_size: int, bits: int) -> List[str]:
    """Crea la población inicial aleatoria de cromosomas binarios."""
    # Generamos 'pop_size' cromosomas. Cada cromosoma es una cadena de 'bits' caracteres
    # donde cada carácter es '0' o '1' elegido al azar.
    return ["".join(random.choice("01") for _ in range(bits)) for _ in range(pop_size)]


def fitness_values(values: List[float], optimize_max: bool) -> List[float]:
    """Calcula los valores de aptitud a partir de los valores de la función según el tipo de optimización."""
    # En un algoritmo genético, la "aptitud" (fitness) indica qué tan "bueno" es un individuo.
    # Aquí partimos de los valores F(x) ya calculados y los convertimos en aptitudes positivas.
    if optimize_max:
        # Si queremos MAXIMIZAR F(x), el que tenga F(x) más grande debe tener más aptitud.
        min_val = min(values)
        # Restamos el mínimo para que la peor no quede negativa y sumamos un epsilon (1e-6)
        # para evitar valores cero exactos.
        return [v - min_val + 1e-6 for v in values]
    else:
        # Si queremos MINIMIZAR F(x), el que tenga F(x) más chico debe tener más aptitud.
        max_val = max(values)
        return [max_val - v + 1e-6 for v in values]


def select(population: List[str], fitness: List[float]) -> Tuple[str, str]:
    """Selecciona dos padres utilizando selección por ruleta."""
    # 'random.choices' permite elegir elementos con probabilidades proporcionales a 'weights'.
    # Así, los cromosomas con mayor aptitud tienen más chances de ser seleccionados como padres.
    return tuple(random.choices(population, weights=fitness, k=2))


def crossover(parent1: str, parent2: str, rate: float = 0.7) -> Tuple[str, str]:
    """Realiza cruza de un punto entre dos padres."""
    # La "cruza" (crossover) mezcla los genes de los padres para producir hijos.
    # Con probabilidad 'rate', cortamos ambas cadenas en un punto al azar y cruzamos las colas.
    if random.random() < rate:
        point = random.randint(1, len(parent1) - 1)  # el punto de corte no puede ser 0 ni el final
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    # Si no hay cruza (por azar), devolvemos copias idénticas de los padres.
    return parent1, parent2


def mutate(bitstring: str, rate: float = 0.01) -> str:
    """Invierte bits en el cromosoma con la probabilidad de mutación dada."""
    # La "mutación" introduce cambios pequeños y aleatorios que ayudan a explorar mejor el espacio.
    # Recorremos cada bit y, con probabilidad 'rate', lo invertimos (0 -> 1, 1 -> 0).
    new_bits = [
        "1" if (bit == "0" and random.random() < rate) else
        "0" if (bit == "1" and random.random() < rate) else
        bit
        for bit in bitstring
    ]
    return "".join(new_bits)


def find_all_extrema(
    coefficients: List[float],
    lower: float,
    upper: float,
    optimize_max: bool,
) -> List[Tuple[float, float]]:
    """Calcula todos los extremos locales del polinomio en el intervalo dado."""

    # Creamos un objeto polinómico de numpy (nota: invierte el orden de coeficientes)
    # 'np.poly1d' espera coeficientes desde el mayor grado al menor, por eso los 'reversed'.
    poly = np.poly1d(list(reversed(coefficients)))
    d1 = poly.deriv()  # primera derivada
    d2 = d1.deriv()    # segunda derivada

    extrema: List[Tuple[float, float]] = []
    # Raíces de la primera derivada (d1.r) son candidatos a máximos/mínimos locales.
    for root in d1.r:
        # Solo consideramos raíces reales (las imaginarias aparecen por redondeos numéricos).
        if abs(root.imag) < 1e-8:
            x = root.real
            # También deben caer dentro del intervalo [lower, upper].
            if lower <= x <= upper:
                second = d2(x)  # evaluamos la segunda derivada en x
                fx = poly(x)    # y el valor del polinomio original en x
                # Test de la segunda derivada:
                if optimize_max and second < 0:
                    # Si buscamos máximos y la segunda derivada es negativa, es un máximo local.
                    extrema.append((x, fx))
                elif not optimize_max and second > 0:
                    # Si buscamos mínimos y la segunda derivada es positiva, es un mínimo local.
                    extrema.append((x, fx))
    # Ordenamos los extremos por su posición en x para imprimir prolijo.
    return sorted(extrema, key=lambda t: t[0])


# ------ NÚCLEO DEL ALGORITMO GENÉTICO ------

def genetic_algorithm(
    coefficients: List[float],
    lower: float,
    upper: float,
    optimize_max: bool,
    bits: int,
    pop_size: int,
    generations: int,
) -> Tuple[List[float], List[float], Tuple[str, float, float], List[float]]:
    """Ejecuta el algoritmo genético y devuelve la población final y los extremos.

    Devuelve una tupla que contiene:
    - lista de valores de x decodificados de la población final,
    - lista de valores de F(x) correspondientes,
    - una tupla con (mejor_cromosoma, mejor_x, mejor_fx),
    - historial del mejor valor de la función por generación para graficar.
    """
    # 1) Empezamos creando una población inicial de cromosomas aleatorios
    population = initial_population(pop_size, bits)

    # 2) Elegimos como "mejor" solución inicial el primer individuo, decodificándolo a x
    best_chromosome = population[0]
    best_x = decode(best_chromosome, lower, upper, bits)
    best_fx = polynomial_value(best_x, coefficients)

    # 3) 'history' guardará, en cada generación, el mejor valor F(x) encontrado hasta ese momento.
    history = [best_fx]

    # 4) Bucle principal: repetimos el proceso 'generations' veces.
    for _ in range(generations):
        # 4.a) Decodificar todos los cromosomas a números reales x y evaluar F(x)
        decoded = [decode(ch, lower, upper, bits) for ch in population]
        values  = [polynomial_value(x, coefficients) for x in decoded]

        # 4.b) Actualizar la mejor solución si encontramos una más conveniente
        for ch, x_val, fx in zip(population, decoded, values):
            if (optimize_max and fx > best_fx) or (not optimize_max and fx < best_fx):
                best_chromosome, best_x, best_fx = ch, x_val, fx

        # Guardamos el mejor valor de esta generación en la historia (para graficar luego)
        history.append(best_fx)

        # 4.c) Calcular aptitudes (fitness) a partir de los valores F(x)
        fitness = fitness_values(values, optimize_max)

        # 4.d) Crear la siguiente generación usando selección, cruza y mutación
        next_generation = []
        while len(next_generation) < pop_size:
            # Seleccionamos dos padres al azar ponderado (ruleta)
            parent1, parent2 = select(population, fitness)
            # Aplicamos cruza para mezclar información genética
            child1, child2 = crossover(parent1, parent2)
            # Mutamos levemente a los hijos para mantener diversidad
            child1 = mutate(child1)
            child2 = mutate(child2)
            # Agregamos a la nueva población
            next_generation.extend([child1, child2])
        # Recortamos por si nos pasamos de tamaño
        population = next_generation[:pop_size]

    # 5) Al terminar, decodificamos la población final para mostrarla y la evaluamos una vez más.
    final_decoded = [decode(ch, lower, upper, bits) for ch in population]
    final_values  = [polynomial_value(x, coefficients) for x in final_decoded]

    # Devolvemos todo lo necesario para imprimir resultados y graficar.
    return final_decoded, final_values, (best_chromosome, best_x, best_fx), history


# ------ PROGRAMA PRINCIPAL (INTERFAZ DE CONSOLA) ------

def main() -> None:
    """Interfaz interactiva de línea de comandos para configurar y ejecutar el algoritmo."""
    print("Optimización de polinomios mediante un Algoritmo Genético\n")

    # Pedimos al usuario que ingrese los parámetros uno por uno.
    # 'input' lee como texto; luego convertimos a int o float según corresponda.
    degree = int(input("Grado del polinomio: "))
    coeffs_input = input(
        f"Coeficientes a0 .. a{degree} separados por espacios: "
    )
    # Convertimos la cadena separada por espacios a una lista de flotantes
    coefficients = [float(c) for c in coeffs_input.split()]
    # Validación: si el usuario dijo grado = n, debe haber exactamente n+1 coeficientes (a0..an)
    if len(coefficients) != degree + 1:
        raise ValueError("El número de coeficientes no coincide con el grado indicado")

    # 'flag' define si optimizamos para mínimo (0) o máximo (1)
    flag = int(input("Bandera (0 = minimizar, 1 = maximizar): "))
    optimize_max = flag == 1

    # Intervalo de búsqueda para x: [lower, upper]
    lower = float(input("Extremo inferior del intervalo de búsqueda (b): "))
    upper = float(input("Extremo superior del intervalo de búsqueda (c): "))

    # Parámetros del algoritmo genético: bits por cromosoma, tamaño de población y generaciones.
    bits = int(input("Número de bits por cromosoma: "))
    pop_size = int(input("Tamaño de la población: "))
    generations = int(input("Número máximo de generaciones: "))

    # Ejecutamos el algoritmo con lo que el usuario especificó.
    final_pop, final_vals, best, history = genetic_algorithm(
        coefficients, lower, upper, optimize_max, bits, pop_size, generations
    )

    # Desempaquetamos la mejor solución para imprimirla claro.
    best_chromosome, best_x, best_fx = best

    # Mostramos la población final (los valores reales de x de cada cromosoma final)
    print("\nPoblación final (valores de x):")
    for val in final_pop:
        print(f"{val:.6f}")

    # También buscamos y listamos los extremos locales teóricos (usando cálculo con derivadas).
    print("\nExtremos locales detectados:")
    extrema = find_all_extrema(coefficients, lower, upper, optimize_max)
    for x, fx in extrema:
        print(f"x = {x:.6f}, F(x) = {fx:.6f}")

    # Y destacamos la mejor solución que encontró el algoritmo genético.
    print("\nMejor cromosoma:", best_chromosome)
    print(f"Mejor x = {best_x:.6f}")
    print(f"F(x) = {best_fx:.6f}")

    # Si 'matplotlib' está disponible, generamos y guardamos un gráfico 'convergence.png'
    # que muestra cómo va mejorando el mejor valor a lo largo de las generaciones.
    if plt is not None:
        plt.plot(history)
        plt.title("Convergencia del algoritmo genético")
        plt.xlabel("Generación")
        plt.ylabel("Mejor F(x)")
        plt.grid(True)
        plt.savefig("convergence.png")
        print("\nGráfico de convergencia guardado como 'convergence.png'.")
    else:
        print("\nMatplotlib no está disponible; no se generó gráfico de convergencia.")


# Este condicional asegura que 'main()' se ejecute solo cuando corremos este archivo directamente
# (y no cuando lo importamos desde otro archivo como un módulo).
if __name__ == "__main__":
    main()
