"""Punto de entrada principal para la optimización de polinomios mediante un algoritmo genético."""

import random
from typing import List, Tuple

# La librería matplotlib se utiliza para graficar la convergencia. Se intenta
# importar, pero si no está disponible el programa continúa sin generar el
# gráfico para facilitar la ejecución en entornos limitados.
try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - solo se ejecuta si falta la librería
    plt = None


def polynomial_value(x: float, coefficients: List[float]) -> float:
    """Devuelve F(x) para un polinomio definido por sus coeficientes.

    coefficients[i] corresponde al coeficiente a_i de x**i.
    El polinomio se evalúa utilizando una suma simple.
    """
    # Itera sobre la lista de coeficientes generando a * x**i para cada término
    return sum(coef * (x ** i) for i, coef in enumerate(coefficients))


def decode(bitstring: str, lower: float, upper: float, bits: int) -> float:
    """Traduce un cromosoma binario a un valor real de x dentro de [lower, upper]."""
    # Mayor entero representable con el número de bits dado
    max_int = 2 ** bits - 1
    # Convierte el cromosoma de base 2 a un entero
    integer_value = int(bitstring, 2)
    # Escala el entero al intervalo deseado [lower, upper]
    return lower + (integer_value / max_int) * (upper - lower)


def encode(x: float, lower: float, upper: float, bits: int) -> str:
    """Convierte un valor real x en su representación binaria de cromosoma."""
    max_int = 2 ** bits - 1
    # Normaliza x a un valor entre 0 y 1 y escala al rango entero
    scaled = int(round((x - lower) / (upper - lower) * max_int))
    # Formatea el entero como una cadena binaria rellenada con ceros a la izquierda
    return format(scaled, f"0{bits}b")


def initial_population(pop_size: int, bits: int) -> List[str]:
    """Crea la población inicial aleatoria de cromosomas binarios."""
    # Cada cromosoma es una cadena de bits escogidos aleatoriamente
    return ["".join(random.choice("01") for _ in range(bits)) for _ in range(pop_size)]


def fitness_values(values: List[float], optimize_max: bool) -> List[float]:
    """Calcula los valores de aptitud a partir de los valores de la función según el tipo de optimización."""
    # La aptitud debe ser positiva. Desplazamos los valores para que el peor tenga baja aptitud.
    if optimize_max:
        min_val = min(values)
        # Valores de la función más altos => mayor aptitud
        return [v - min_val + 1e-6 for v in values]  # un pequeño epsilon evita aptitud cero
    else:
        max_val = max(values)
        # Valores de la función más bajos => mayor aptitud para minimización
        return [max_val - v + 1e-6 for v in values]


def select(population: List[str], fitness: List[float]) -> Tuple[str, str]:
    """Selecciona dos padres utilizando selección por ruleta."""
    # random.choices realiza la selección proporcionalmente a los pesos proporcionados
    return tuple(random.choices(population, weights=fitness, k=2))


def crossover(parent1: str, parent2: str, rate: float = 0.7) -> Tuple[str, str]:
    """Realiza cruza de un punto entre dos padres."""
    # Con la tasa de cruza, intercambia las colas de los padres después de un punto aleatorio
    if random.random() < rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    # Si no ocurre cruza, los descendientes son copias exactas de los padres
    return parent1, parent2


def mutate(bitstring: str, rate: float = 0.01) -> str:
    """Invierte bits en el cromosoma con la probabilidad de mutación dada."""
    # Itera por cada bit y posiblemente lo invierte
    new_bits = [
        "1" if (bit == "0" and random.random() < rate) else
        "0" if (bit == "1" and random.random() < rate) else
        bit
        for bit in bitstring
    ]
    return "".join(new_bits)


def genetic_algorithm(
    coefficients: List[float],
    lower: float,
    upper: float,
    optimize_max: bool,
    bits: int,
    pop_size: int,
    generations: int,
) -> Tuple[List[float], Tuple[str, float, float], List[float]]:
    """Ejecuta el algoritmo genético y devuelve la población, el mejor individuo y el historial.

    Devuelve una tupla que contiene:
    - lista de valores de x decodificados de la población final
    - una tupla con (mejor_cromosoma, mejor_x, mejor_fx)
    - historial del mejor valor de la función por generación para graficar
    """
    population = initial_population(pop_size, bits)
    # Inicializa la mejor solución con el primer individuo
    best_chromosome = population[0]
    best_x = decode(best_chromosome, lower, upper, bits)
    best_fx = polynomial_value(best_x, coefficients)
    # La historia guarda el mejor valor de cada generación para el gráfico de convergencia
    history = [best_fx]

    for _ in range(generations):
        # Decodifica y evalúa la población
        # Decodifica los cromosomas en números reales y evalúa el polinomio
        decoded = [decode(ch, lower, upper, bits) for ch in population]
        values = [polynomial_value(x, coefficients) for x in decoded]

        # Actualiza la mejor solución
        for ch, x_val, fx in zip(population, decoded, values):
            if (optimize_max and fx > best_fx) or (not optimize_max and fx < best_fx):
                best_chromosome, best_x, best_fx = ch, x_val, fx

        history.append(best_fx)

        # Calcula la aptitud y crea la siguiente generación
        fitness = fitness_values(values, optimize_max)
        next_generation = []
        # Crea una nueva población completa
        while len(next_generation) < pop_size:
            # Selecciona padres proporcionalmente a su aptitud
            parent1, parent2 = select(population, fitness)
            # Aplica cruza y mutación para producir descendencia
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation.extend([child1, child2])
        population = next_generation[:pop_size]

    final_decoded = [decode(ch, lower, upper, bits) for ch in population]
    return final_decoded, (best_chromosome, best_x, best_fx), history


def main() -> None:
    """Interfaz interactiva de línea de comandos para configurar y ejecutar el algoritmo."""
    print("Optimización de polinomios mediante un Algoritmo Genético\n")

    # Reúne los parámetros del usuario
    degree = int(input("Grado del polinomio: "))
    coeffs_input = input(
        f"Coeficientes a0 .. a{degree} separados por espacios: "
    )
    coefficients = [float(c) for c in coeffs_input.split()]
    if len(coefficients) != degree + 1:
        raise ValueError("El número de coeficientes no coincide con el grado indicado")

    flag = int(input("Bandera (0 = minimizar, 1 = maximizar): "))
    optimize_max = flag == 1

    lower = float(input("Extremo inferior del intervalo de búsqueda (b): "))
    upper = float(input("Extremo superior del intervalo de búsqueda (c): "))

    bits = int(input("Número de bits por cromosoma: "))
    pop_size = int(input("Tamaño de la población: "))
    generations = int(input("Número máximo de generaciones: "))

    # Ejecuta el algoritmo genético con los parámetros proporcionados
    final_pop, best, history = genetic_algorithm(
        coefficients, lower, upper, optimize_max, bits, pop_size, generations
    )

    # Muestra los resultados
    best_chromosome, best_x, best_fx = best
    print("\nPoblación final (valores de x):")
    for val in final_pop:
        print(f"{val:.6f}")

    print("\nMejor cromosoma:", best_chromosome)
    print(f"Mejor x = {best_x:.6f}")
    print(f"F(x) = {best_fx:.6f}")

    # Grafica la convergencia y guarda la figura si matplotlib está disponible
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


if __name__ == "__main__":
    main()