# ==============================================
# VERSIÓN GUIADA PARA ESTUDIO E IMPRESIÓN
# ==============================================
# Este archivo es el mismo que ``main.py`` pero con muchos comentarios
# que explican cada paso.

"""Programa para optimizar polinomios mediante un algoritmo genético."""

# ------ LIBRERÍAS Y TIPOS BÁSICOS ------
# 'random' sirve para generar números aleatorios, necesarios en la simulación genética.
import random
# 'typing' nos permite aclarar los tipos de datos que usa cada función.
from typing import List, Tuple

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


def cruzar_cromosomas(padre1: str, padre2: str) -> Tuple[str, str]:
    """Mezcla la información de dos padres para producir dos hijos siempre nuevos."""

    # La longitud del cromosoma se usa para fijar los límites del punto de corte
    longitud = len(padre1)
    # Calculamos el 25 % y el 75 % de la longitud. El punto de cruza debe
    # quedar en ese rango intermedio para que ambos padres aporten información.
    limite_inferior = max(1, int(longitud * 0.25))
    limite_superior = min(longitud - 1, int(longitud * 0.75))
    # En cromosomas muy cortos, el 25 % y el 75 % pueden coincidir; lo ajustamos.
    if limite_superior < limite_inferior:
        limite_superior = limite_inferior
    # Elegimos aleatoriamente el punto de cruza dentro de los límites calculados
    punto_cruza = random.randint(limite_inferior, limite_superior)
    # Construimos los hijos: la primera parte proviene de un padre y la segunda del otro
    hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
    hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]
    # Devolvemos las dos nuevas combinaciones genéticas
    return hijo1, hijo2


def mutar_cromosoma(
    cadena_bits: str, probabilidad_mutacion: float = 0.01, max_bits: int = 10
) -> str:
    """Realiza mutaciones en hasta ``max_bits`` posiciones elegidas al azar."""

    # Elegimos sin repetición las posiciones candidatas a mutar. Si el cromosoma
    # tiene menos de 'max_bits' genes, tomamos todos los posibles.
    indices = random.sample(range(len(cadena_bits)), k=min(max_bits, len(cadena_bits)))
    # Convertimos la cadena a lista para poder modificar posiciones individuales
    bits = list(cadena_bits)
    for i in indices:
        # Para cada índice seleccionado, decidimos si muta según 'probabilidad_mutacion'
        if random.random() < probabilidad_mutacion:
            # Invertimos el bit: 0 -> 1 o 1 -> 0
            bits[i] = "1" if bits[i] == "0" else "0"
    # Reconstruimos la cadena de bits mutada
    return "".join(bits)



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

