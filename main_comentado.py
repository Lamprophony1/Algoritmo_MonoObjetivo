# ==============================================
# VERSIÓN GUIADA PARA ESTUDIO E IMPRESIÓN
# ==============================================
# Este archivo es el mismo que ``main.py`` pero con muchos comentarios
# que explican cada paso. Se pensó para personas que están aprendiendo
# programación, por lo que se intenta usar un lenguaje claro y amigable.

"""Programa para optimizar polinomios mediante un algoritmo genético."""

# ------ LIBRERÍAS Y TIPOS BÁSICOS ------
# 'random' sirve para generar números aleatorios, necesarios en la simulación genética.
import random
# 'typing' nos permite aclarar los tipos de datos que usa cada función.
from typing import List, Tuple

# 'numpy' ofrece utilidades matemáticas. Aquí lo usamos para derivar el polinomio
# y encontrar sus extremos mediante cálculo.
import numpy as np

# 'matplotlib' es opcional y solo se utiliza para dibujar un gráfico de convergencia.
# Si no está instalada, el programa sigue funcionando sin el gráfico.
try:  # pragma: no cover - solo se ejecuta si la librería no está
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - cualquier error al importar la deja como None
    plt = None


# ------ FUNCIONES AUXILIARES SOBRE EL POLINOMIO ------

def evaluar_polinomio(valor_x: float, coeficientes: List[float]) -> float:
    """Calcula F(x) para un polinomio definido por sus coeficientes.

    ``coeficientes[i]`` es el coeficiente del término ``x**i``.
    """
    # Construimos el valor del polinomio sumando cada término: a0 + a1*x + a2*x^2 + ...
    return sum(
        coeficiente * (valor_x ** potencia)
        for potencia, coeficiente in enumerate(coeficientes)
    )


def decodificar_cromosoma(
    cadena_bits: str, limite_inferior: float, limite_superior: float, cantidad_bits: int
) -> float:
    """Traduce una cadena de bits a un valor real del intervalo dado."""

    # Interpretamos la cadena como un número entero en base 2
    maximo_entero = 2 ** cantidad_bits - 1
    valor_entero = int(cadena_bits, 2)
    # Luego lo escalamos para que se ubique entre 'limite_inferior' y 'limite_superior'
    return limite_inferior + (valor_entero / maximo_entero) * (
        limite_superior - limite_inferior
    )


def codificar_valor(
    valor_x: float, limite_inferior: float, limite_superior: float, cantidad_bits: int
) -> str:
    """Hace la operación inversa: pasa de un valor real a una cadena de bits."""

    maximo_entero = 2 ** cantidad_bits - 1
    valor_escalado = int(
        round((valor_x - limite_inferior) / (limite_superior - limite_inferior) * maximo_entero)
    )
    # format(..., f"0{cantidad_bits}b") genera la representación binaria rellenada con ceros
    return format(valor_escalado, f"0{cantidad_bits}b")


def generar_poblacion_inicial(tamano_poblacion: int, cantidad_bits: int) -> List[str]:
    """Genera la primera población aleatoria de cromosomas."""

    # Cada cromosoma es simplemente una cadena de bits al azar
    return [
        "".join(random.choice("01") for _ in range(cantidad_bits))
        for _ in range(tamano_poblacion)
    ]


def calcular_aptitudes(valores_funcion: List[float], optimizar_max: bool) -> List[float]:
    """Asigna una 'aptitud' positiva a cada valor de F(x)."""

    if optimizar_max:
        # Para maximizar: los valores grandes de F(x) deben tener más aptitud
        valor_minimo = min(valores_funcion)
        return [v - valor_minimo + 1e-6 for v in valores_funcion]
    # Para minimizar: los valores pequeños de F(x) deben tener más aptitud
    valor_maximo = max(valores_funcion)
    return [valor_maximo - v + 1e-6 for v in valores_funcion]


def seleccionar_padres(poblacion: List[str], aptitudes: List[float]) -> Tuple[str, str]:
    """Elige dos padres usando la técnica de 'ruleta'. Los más aptos tienen más chances."""

    return tuple(random.choices(poblacion, weights=aptitudes, k=2))


def cruzar_cromosomas(
    padre1: str, padre2: str, probabilidad_cruza: float = 0.7
) -> Tuple[str, str]:
    """Mezcla la información de dos padres para producir dos hijos."""

    if random.random() < probabilidad_cruza:
        # Elegimos al azar un punto de corte que no sea extremo
        punto_cruza = random.randint(1, len(padre1) - 1)
        hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
        hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]
        return hijo1, hijo2
    # Si no ocurre cruza, los hijos son copias de los padres
    return padre1, padre2


def mutar_cromosoma(cadena_bits: str, probabilidad_mutacion: float = 0.01) -> str:
    """Realiza pequeñas modificaciones aleatorias (mutaciones) en un cromosoma."""

    nuevos_bits = [
        "1" if (bit == "0" and random.random() < probabilidad_mutacion) else
        "0" if (bit == "1" and random.random() < probabilidad_mutacion) else
        bit
        for bit in cadena_bits
    ]
    return "".join(nuevos_bits)


def encontrar_extremos(
    coeficientes: List[float],
    limite_inferior: float,
    limite_superior: float,
    optimizar_max: bool,
) -> List[Tuple[float, float]]:
    """Busca los puntos donde el polinomio tiene máximos o mínimos locales."""

    polinomio = np.poly1d(list(reversed(coeficientes)))
    primera_derivada = polinomio.deriv()
    segunda_derivada = primera_derivada.deriv()
    lista_extremos: List[Tuple[float, float]] = []
    for raiz in primera_derivada.r:  # Analizamos las raíces de la primera derivada
        if abs(raiz.imag) < 1e-8:  # Descartamos raíces complejas
            valor_x = raiz.real
            if limite_inferior <= valor_x <= limite_superior:
                segunda_eval = segunda_derivada(valor_x)
                valor_fx = polinomio(valor_x)
                if optimizar_max and segunda_eval < 0:
                    lista_extremos.append((valor_x, valor_fx))
                elif not optimizar_max and segunda_eval > 0:
                    lista_extremos.append((valor_x, valor_fx))
    return sorted(lista_extremos, key=lambda t: t[0])


# ------ ALGORITMO GENÉTICO COMPLETO ------

def algoritmo_genetico(
    coeficientes: List[float],
    limite_inferior: float,
    limite_superior: float,
    optimizar_max: bool,
    cantidad_bits: int,
    tamano_poblacion: int,
    numero_generaciones: int,
) -> Tuple[List[float], List[float], Tuple[str, float, float], List[float]]:
    """Ejecuta todas las etapas del algoritmo genético y devuelve información útil."""

    # 1) Comenzamos con una población aleatoria
    poblacion = generar_poblacion_inicial(tamano_poblacion, cantidad_bits)

    # 2) Suponemos que el mejor individuo inicial es el primero
    mejor_cromosoma = poblacion[0]
    mejor_valor_x = decodificar_cromosoma(
        mejor_cromosoma, limite_inferior, limite_superior, cantidad_bits
    )
    mejor_valor_fx = evaluar_polinomio(mejor_valor_x, coeficientes)

    # 3) Guardaremos el mejor valor de cada generación para luego graficarlo
    historial_mejores = [mejor_valor_fx]

    # 4) Repetimos el proceso tantas generaciones como se haya pedido
    for _ in range(numero_generaciones):
        # 4.a) Decodificamos cada cromosoma a su valor real y evaluamos el polinomio
        valores_x = [
            decodificar_cromosoma(crom, limite_inferior, limite_superior, cantidad_bits)
            for crom in poblacion
        ]
        valores_fx = [evaluar_polinomio(vx, coeficientes) for vx in valores_x]

        # 4.b) Actualizamos el mejor individuo si encontramos uno superior
        for crom, valor_x_temp, valor_fx_temp in zip(poblacion, valores_x, valores_fx):
            if (optimizar_max and valor_fx_temp > mejor_valor_fx) or (
                not optimizar_max and valor_fx_temp < mejor_valor_fx
            ):
                mejor_cromosoma = crom
                mejor_valor_x = valor_x_temp
                mejor_valor_fx = valor_fx_temp

        historial_mejores.append(mejor_valor_fx)

        # 4.c) Calculamos aptitudes y creamos la nueva generación
        aptitudes = calcular_aptitudes(valores_fx, optimizar_max)
        siguiente_generacion = []
        while len(siguiente_generacion) < tamano_poblacion:
            # Seleccionamos dos padres
            padre1, padre2 = seleccionar_padres(poblacion, aptitudes)
            # Cruzamos y luego mutamos para crear dos hijos
            hijo1, hijo2 = cruzar_cromosomas(padre1, padre2)
            hijo1 = mutar_cromosoma(hijo1)
            hijo2 = mutar_cromosoma(hijo2)
            siguiente_generacion.extend([hijo1, hijo2])
        poblacion = siguiente_generacion[:tamano_poblacion]

    # 5) Decodificamos la población final para mostrar los valores reales
    poblacion_decodificada = [
        decodificar_cromosoma(crom, limite_inferior, limite_superior, cantidad_bits)
        for crom in poblacion
    ]
    valores_finales = [evaluar_polinomio(vx, coeficientes) for vx in poblacion_decodificada]

    return poblacion_decodificada, valores_finales, (
        mejor_cromosoma,
        mejor_valor_x,
        mejor_valor_fx,
    ), historial_mejores


# ------ PROGRAMA PRINCIPAL (INTERFAZ DE CONSOLA) ------

def programa_principal() -> None:
    """Dialoga con el usuario para obtener parámetros y muestra los resultados."""

    print("Optimización de polinomios mediante un Algoritmo Genético\n")

    # Solicitamos datos del polinomio
    grado_polinomio = int(input("Grado del polinomio: "))
    entrada_coeficientes = input(
        f"Coeficientes a0 .. a{grado_polinomio} separados por espacios: "
    )
    coeficientes = [float(c) for c in entrada_coeficientes.split()]
    if len(coeficientes) != grado_polinomio + 1:
        raise ValueError(
            "El número de coeficientes no coincide con el grado indicado"
        )

    # Indicamos si queremos maximizar o minimizar F(x)
    bandera = int(input("Bandera (0 = minimizar, 1 = maximizar): "))
    optimizar_max = bandera == 1

    # Intervalo de búsqueda
    limite_inferior = float(
        input("Extremo inferior del intervalo de búsqueda (b): ")
    )
    limite_superior = float(
        input("Extremo superior del intervalo de búsqueda (c): ")
    )

    # Parámetros del algoritmo genético
    cantidad_bits = int(input("Número de bits por cromosoma: "))
    tamano_poblacion = int(input("Tamaño de la población: "))
    numero_generaciones = int(input("Número máximo de generaciones: "))

    # Ejecutamos el algoritmo
    poblacion_final, valores_finales, mejor_resultado, historial_mejores = algoritmo_genetico(
        coeficientes,
        limite_inferior,
        limite_superior,
        optimizar_max,
        cantidad_bits,
        tamano_poblacion,
        numero_generaciones,
    )

    mejor_cromosoma, mejor_valor_x, mejor_valor_fx = mejor_resultado

    # Mostramos la población final en términos reales
    print("\nPoblación final (valores de x):")
    for valor in poblacion_final:
        print(f"{valor:.6f}")

    # Listamos extremos locales calculados analíticamente
    print("\nExtremos locales detectados:")
    extremos = encontrar_extremos(
        coeficientes, limite_inferior, limite_superior, optimizar_max
    )
    for valor_x, valor_fx in extremos:
        print(f"x = {valor_x:.6f}, F(x) = {valor_fx:.6f}")

    # Resaltamos el mejor individuo encontrado por el algoritmo
    print("\nMejor cromosoma:", mejor_cromosoma)
    print(f"Mejor x = {mejor_valor_x:.6f}")
    print(f"F(x) = {mejor_valor_fx:.6f}")

    # Graficamos la evolución si es posible
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


# Este condicional evita que se ejecute automáticamente cuando se importa como módulo
if __name__ == "__main__":
    programa_principal()

