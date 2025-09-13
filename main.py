"""Programa para optimizar polinomios mediante un algoritmo genético."""

import random
from typing import List, Tuple


# La librería matplotlib se utiliza solo para el gráfico de convergencia.  Si no
# está instalada, el programa continúa funcionando sin generar dicho gráfico.
try:  # pragma: no cover - solo se ejecuta si falta la librería
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - manejamos cualquier problema de importación
    plt = None


def evaluar_polinomio(valor_x: float, coeficientes: List[float]) -> float:
    """Devuelve el valor F(x) de un polinomio para un valor de ``x``.

    ``coeficientes[i]`` es el coeficiente :math:`a_i` del término ``x**i``.
    """

    return sum(
        coeficiente * (valor_x ** potencia)
        for potencia, coeficiente in enumerate(coeficientes)
    )


def decodificar_cromosoma(
    cadena_bits: str, limite_inferior: float, limite_superior: float, cantidad_bits: int
) -> float:
    """Convierte un cromosoma binario en un valor real dentro del intervalo dado."""

    maximo_entero = 2 ** cantidad_bits - 1
    valor_entero = int(cadena_bits, 2)
    return limite_inferior + (valor_entero / maximo_entero) * (
        limite_superior - limite_inferior
    )


def codificar_valor(
    valor_x: float, limite_inferior: float, limite_superior: float, cantidad_bits: int
) -> str:
    """Convierte un valor real ``x`` a su representación binaria."""

    maximo_entero = 2 ** cantidad_bits - 1
    valor_escalado = int(
        round((valor_x - limite_inferior) / (limite_superior - limite_inferior) * maximo_entero)
    )
    return format(valor_escalado, f"0{cantidad_bits}b")


def generar_poblacion_inicial(tamano_poblacion: int, cantidad_bits: int) -> List[str]:
    """Crea la población inicial de cromosomas binarios."""

    return [
        "".join(random.choice("01") for _ in range(cantidad_bits))
        for _ in range(tamano_poblacion)
    ]


def calcular_aptitudes(valores_funcion: List[float], optimizar_max: bool) -> List[float]:
    """Calcula la aptitud de cada individuo según los valores de ``F(x)``."""

    if optimizar_max:
        valor_minimo = min(valores_funcion)
        return [v - valor_minimo + 1e-6 for v in valores_funcion]
    valor_maximo = max(valores_funcion)
    return [valor_maximo - v + 1e-6 for v in valores_funcion]


def seleccionar_padres(poblacion: List[str], aptitudes: List[float]) -> Tuple[str, str]:
    """Elige dos padres mediante selección por ruleta."""

    return tuple(random.choices(poblacion, weights=aptitudes, k=2))


def cruzar_cromosomas(padre1: str, padre2: str) -> Tuple[str, str]:
    """Realiza una cruza de un punto entre dos padres siempre generando nuevos hijos.

    El punto de corte se elige aleatoriamente entre el 25 % y el 75 % de la
    longitud del cromosoma para garantizar que ambos fragmentos aporten
    información de cada progenitor.
    """

    longitud = len(padre1)
    limite_inferior = max(1, int(longitud * 0.25))
    limite_superior = min(longitud - 1, int(longitud * 0.75))
    if limite_superior < limite_inferior:
        limite_superior = limite_inferior
    punto_cruza = random.randint(limite_inferior, limite_superior)
    hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
    hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]
    return hijo1, hijo2


def mutar_cromosoma(
    cadena_bits: str, probabilidad_mutacion: float = 0.01, max_bits: int = 10
) -> str:
    """Invierte como máximo ``max_bits`` bits aleatorios del cromosoma."""

    indices = random.sample(range(len(cadena_bits)), k=min(max_bits, len(cadena_bits)))
    bits = list(cadena_bits)
    for i in indices:
        if random.random() < probabilidad_mutacion:
            bits[i] = "1" if bits[i] == "0" else "0"
    return "".join(bits)


def algoritmo_genetico(
    coeficientes: List[float],
    limite_inferior: float,
    limite_superior: float,
    optimizar_max: bool,
    cantidad_bits: int,
    tamano_poblacion: int,
    numero_generaciones: int,
) -> Tuple[
    List[float], List[float], List[Tuple[str, float, float]], List[float]
]:
    """Ejecuta el algoritmo genético y devuelve resultados detallados."""

    poblacion = generar_poblacion_inicial(tamano_poblacion, cantidad_bits)

    mejores: List[Tuple[str, float, float]] = []

    def actualizar_mejores(crom: str, valor_x: float, valor_fx: float) -> None:
        mejores.append((crom, valor_x, valor_fx))
        if optimizar_max:
            mejores.sort(key=lambda t: t[2], reverse=True)
        else:
            mejores.sort(key=lambda t: t[2])
        del mejores[2:]

    valores_x = [
        decodificar_cromosoma(crom, limite_inferior, limite_superior, cantidad_bits)
        for crom in poblacion
    ]
    valores_fx = [evaluar_polinomio(vx, coeficientes) for vx in valores_x]
    for crom, valor_x, valor_fx in zip(poblacion, valores_x, valores_fx):
        actualizar_mejores(crom, valor_x, valor_fx)

    historial_mejores = [mejores[0][2]]

    for _ in range(numero_generaciones):
        aptitudes = calcular_aptitudes(valores_fx, optimizar_max)
        siguiente_generacion = []
        while len(siguiente_generacion) < tamano_poblacion:
            padre1, padre2 = seleccionar_padres(poblacion, aptitudes)
            hijo1, hijo2 = cruzar_cromosomas(padre1, padre2)
            hijo1 = mutar_cromosoma(hijo1)
            hijo2 = mutar_cromosoma(hijo2)
            siguiente_generacion.extend([hijo1, hijo2])
        poblacion = siguiente_generacion[:tamano_poblacion]

        valores_x = [
            decodificar_cromosoma(crom, limite_inferior, limite_superior, cantidad_bits)
            for crom in poblacion
        ]
        valores_fx = [evaluar_polinomio(vx, coeficientes) for vx in valores_x]
        for crom, valor_x, valor_fx in zip(poblacion, valores_x, valores_fx):
            actualizar_mejores(crom, valor_x, valor_fx)
        historial_mejores.append(mejores[0][2])

    poblacion_decodificada = valores_x
    valores_finales = valores_fx

    return poblacion_decodificada, valores_finales, mejores, historial_mejores


def programa_principal() -> None:
    """Interfaz de consola para configurar y ejecutar el algoritmo genético."""

    print("Optimización de polinomios mediante un Algoritmo Genético\n")

    grado_polinomio = int(input("Grado del polinomio: "))
    entrada_coeficientes = input(
        f"Coeficientes a0 .. a{grado_polinomio} separados por espacios: "
    )
    coeficientes = [float(c) for c in entrada_coeficientes.split()]
    if len(coeficientes) != grado_polinomio + 1:
        raise ValueError(
            "El número de coeficientes no coincide con el grado indicado"
        )

    bandera = int(input("Bandera (0 = minimizar, 1 = maximizar): "))
    optimizar_max = bandera == 1

    limite_inferior = float(
        input("Extremo inferior del intervalo de búsqueda (b): ")
    )
    limite_superior = float(
        input("Extremo superior del intervalo de búsqueda (c): ")
    )

    cantidad_bits = int(input("Número de bits por cromosoma: "))
    tamano_poblacion = int(input("Tamaño de la población: "))
    numero_generaciones = int(input("Número máximo de generaciones: "))

    poblacion_final, valores_finales, mejores_resultados, historial_mejores = algoritmo_genetico(
        coeficientes,
        limite_inferior,
        limite_superior,
        optimizar_max,
        cantidad_bits,
        tamano_poblacion,
        numero_generaciones,
    )
    print("\nPoblación final (valores de x):")
    for valor in poblacion_final:
        print(f"{valor:.6f}")

    for indice, (_, mejor_valor_x, mejor_valor_fx) in enumerate(
        mejores_resultados, start=1
    ):
        print(f"\nMejor x {indice} = {mejor_valor_x:.6f}")
        print(f"F(x) = {mejor_valor_fx:.6f}")

    if plt is not None:
        plt.plot(historial_mejores)
        plt.title("Convergencia del algoritmo genético")
        plt.xlabel("Generación")
        plt.ylabel("Mejor F(x)")
        plt.grid(True)
        plt.savefig("convergence.png")
        print("\nGráfico de convergencia guardado como 'convergence.png'.")
    else:
        print("\nMatplotlib no está disponible; no se generó gráfico de convergencia.")


if __name__ == "__main__":
    programa_principal()

