"""Experimento de PageRank por cadeia de Markov.

Dependencias: numpy e matplotlib.
Instalacao: pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# Matriz de adjacencia: A[i, j] = 1 quando i aponta para j.
# A pagina 5 nao tem links de saida.
A = np.array(
    [
        [0, 1, 1, 0, 0],  # Pagina 1 -> paginas 2 e 3
        [0, 0, 1, 0, 0],  # Pagina 2 -> pagina 3
        [1, 0, 0, 1, 0],  # Pagina 3 -> paginas 1 e 4
        [0, 0, 1, 0, 1],  # Pagina 4 -> paginas 3 e 5
        [0, 0, 0, 0, 0],  # Pagina 5 -> nenhum link
    ],
    dtype=float,
)


def pagerank(A, alpha=0.85, tol=1e-10, max_iter=10000):
    """Calcula o PageRank pelo metodo das potencias."""
    A = np.asarray(A, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A matriz de adjacencia deve ser quadrada.")
    if np.any(A < 0):
        raise ValueError("A matriz nao pode ter valores negativos.")
    if not 0 < alpha < 1:
        raise ValueError("alpha deve estar entre 0 e 1.")

    n = A.shape[0]
    P = np.zeros((n, n), dtype=float)

    # Construcao da matriz de transicao.
    for i in range(n):
        grau_saida = A[i].sum()
        if grau_saida > 0:
            P[i] = A[i] / grau_saida
        else:
            # Pagina sem saida: distribui a probabilidade igualmente.
            P[i] = 1.0 / n

    # Matriz de transicao com amortecimento.
    G = alpha * P + (1.0 - alpha) / n

    # Distribuicao inicial uniforme.
    pi = np.full(n, 1.0 / n)
    erros = []

    # Metodo das potencias.
    for k in range(1, max_iter + 1):
        novo = pi @ G
        erro = np.linalg.norm(novo - pi, ord=1)
        erros.append(erro)

        if erro < tol:
            return novo, k, G, erros

        pi = novo

    raise RuntimeError("O metodo das potencias nao convergiu.")


def main():
    alphas = [0.50, 0.85, 0.99]
    resultados = {}

    print("\nTABELA 1 - RESULTADOS DO PAGERANK")
    print("alpha | iteracoes |    pi1 |    pi2 |    pi3 |    pi4 |    pi5 | lambda2")
    print("-" * 78)

    for alpha in alphas:
        pi, iteracoes, G, erros = pagerank(A, alpha=alpha)

        # Autovalores em ordem decrescente de modulo.
        modulos = np.sort(np.abs(np.linalg.eigvals(G)))[::-1]
        lambda2 = float(modulos[1])

        resultados[alpha] = {
            "pi": pi,
            "iteracoes": iteracoes,
            "G": G,
            "erros": erros,
            "lambda2": lambda2,
        }

        valores = " | ".join(f"{valor:.4f}" for valor in pi)
        print(f"{alpha:4.2f}  | {iteracoes:9d} | {valores} | {lambda2:.4f}")

        # Conferencias numericas para cada valor de alpha.
        soma = pi.sum()
        residuo = np.linalg.norm(pi - pi @ G, ord=np.inf)
        print(f"       soma das probabilidades = {soma:.12f}; residuo = {residuo:.3e}")

    # Grafico da diferenca entre iteracoes sucessivas.
    for alpha in alphas:
        erros = resultados[alpha]["erros"]
        plt.semilogy(
            range(1, len(erros) + 1),
            erros,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=fr"$\alpha={alpha}$",
        )

    plt.xlabel("Itera\u00e7\u00e3o k")
    plt.ylabel(r"$\|\pi^{(k+1)}-\pi^{(k)}\|_1$")
    plt.title("Converg\u00eancia do m\u00e9todo das pot\u00eancias")
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_convergencia.pdf", bbox_inches="tight")
    plt.close()
    print("\nGrafico salvo em: fig_convergencia.pdf")


if __name__ == "__main__":
    main()
