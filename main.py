"""Main entry point for polynomial optimization using a genetic algorithm."""

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
    """Return F(x) for a polynomial defined by its coefficients.

    coefficients[i] corresponds to the coefficient a_i of x**i.
    The polynomial is evaluated using a simple summation.
    """
    # Iterate over the list of coefficients generating a * x**i for each term
    return sum(coef * (x ** i) for i, coef in enumerate(coefficients))


def decode(bitstring: str, lower: float, upper: float, bits: int) -> float:
    """Translate a binary chromosome into a real x value inside [lower, upper]."""
    # Largest integer representable with the given number of bits
    max_int = 2 ** bits - 1
    # Convert the chromosome from base 2 to an integer
    integer_value = int(bitstring, 2)
    # Scale the integer to the desired interval [lower, upper]
    return lower + (integer_value / max_int) * (upper - lower)


def encode(x: float, lower: float, upper: float, bits: int) -> str:
    """Convert a real value x into its binary chromosome representation."""
    max_int = 2 ** bits - 1
    # Normalize x to a value between 0 and 1 and scale to integer range
    scaled = int(round((x - lower) / (upper - lower) * max_int))
    # Format the integer as a binary string padded with leading zeros
    return format(scaled, f"0{bits}b")


def initial_population(pop_size: int, bits: int) -> List[str]:
    """Create the initial random population of binary chromosomes."""
    # Each chromosome is a string of bits randomly chosen
    return ["".join(random.choice("01") for _ in range(bits)) for _ in range(pop_size)]


def fitness_values(values: List[float], optimize_max: bool) -> List[float]:
    """Compute fitness scores from function values depending on optimization type."""
    # Fitness must be positive. We shift values so that the worst value has low fitness.
    if optimize_max:
        min_val = min(values)
        # Higher function values => higher fitness
        return [v - min_val + 1e-6 for v in values]  # small epsilon avoids zero fitness
    else:
        max_val = max(values)
        # Lower function values => higher fitness for minimization
        return [max_val - v + 1e-6 for v in values]


def select(population: List[str], fitness: List[float]) -> Tuple[str, str]:
    """Select two parents using roulette-wheel selection."""
    # random.choices performs selection proportionally to the provided weights
    return tuple(random.choices(population, weights=fitness, k=2))


def crossover(parent1: str, parent2: str, rate: float = 0.7) -> Tuple[str, str]:
    """Perform single-point crossover between two parents."""
    # With the crossover rate, exchange tails of the parents after a random point
    if random.random() < rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    # If no crossover occurs, offspring are exact copies of the parents
    return parent1, parent2


def mutate(bitstring: str, rate: float = 0.01) -> str:
    """Flip bits in the chromosome with the given mutation probability."""
    # Iterate through each bit and possibly flip it
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
    """Run the genetic algorithm and return population, best individual and history.

    Returns a tuple containing:
    - list of decoded x values of the final population
    - a tuple with (best_chromosome, best_x, best_fx)
    - history of best function value per generation for plotting
    """
    population = initial_population(pop_size, bits)
    # Initialize best solution with first individual
    best_chromosome = population[0]
    best_x = decode(best_chromosome, lower, upper, bits)
    best_fx = polynomial_value(best_x, coefficients)
    # History stores the best value of each generation for the convergence graph
    history = [best_fx]

    for _ in range(generations):
        # Decode and evaluate population
        # Decode chromosomes into real numbers and evaluate the polynomial
        decoded = [decode(ch, lower, upper, bits) for ch in population]
        values = [polynomial_value(x, coefficients) for x in decoded]

        # Update best solution
        for ch, x_val, fx in zip(population, decoded, values):
            if (optimize_max and fx > best_fx) or (not optimize_max and fx < best_fx):
                best_chromosome, best_x, best_fx = ch, x_val, fx

        history.append(best_fx)

        # Calculate fitness and create next generation
        fitness = fitness_values(values, optimize_max)
        next_generation = []
        # Create a full new population
        while len(next_generation) < pop_size:
            # Select parents proportionally to their fitness
            parent1, parent2 = select(population, fitness)
            # Apply crossover and mutation to produce offspring
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1)
            child2 = mutate(child2)
            next_generation.extend([child1, child2])
        population = next_generation[:pop_size]

    final_decoded = [decode(ch, lower, upper, bits) for ch in population]
    return final_decoded, (best_chromosome, best_x, best_fx), history


def main() -> None:
    """Interactive command-line interface to configure and run the algorithm."""
    print("Optimización de polinomios mediante un Algoritmo Genético\n")

    # Collect user parameters
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

    # Run genetic algorithm with the provided parameters
    final_pop, best, history = genetic_algorithm(
        coefficients, lower, upper, optimize_max, bits, pop_size, generations
    )

    # Display results
    best_chromosome, best_x, best_fx = best
    print("\nPoblación final (valores de x):")
    for val in final_pop:
        print(f"{val:.6f}")

    print("\nMejor cromosoma:", best_chromosome)
    print(f"Mejor x = {best_x:.6f}")
    print(f"F(x) = {best_fx:.6f}")

    # Plot convergence and save figure if matplotlib is available
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
