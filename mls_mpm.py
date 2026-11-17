"""
MLS-MPM 2D em Python (NumPy)
=============================

Implementação do núcleo do artigo:
    Y. Hu, Y. Fang, Z. Ge, Z. Qu, Y. Zhu, A. Pradhana, C. Jiang.
    "A Moving Least Squares Material Point Method with Displacement
    Discontinuity and Two-Way Rigid Body Coupling".
    ACM Transactions on Graphics 37(4) — SIGGRAPH 2018.

Simula três materiais com o MESMO algoritmo, mudando apenas o modelo
constitutivo (a "lei" do material):
    água  -> fluido (sem resistência ao cisalhamento)
    geleia -> sólido elástico (corotacional fixo)
    neve  -> elastoplástico com endurecimento (Stomakhin et al. 2013)

Uso:
    python mls_mpm.py                 # janela interativa
    python mls_mpm.py --cena 4        # começa na cena 4
    python mls_mpm.py --salvar demo.gif --quadros 300   # grava sem abrir janela

Controles na janela:
    1-5     troca de cena               espaço  pausa / continua
    r       reinicia a cena             g       mostra a grade (massa nos nós)
    arrastar com o botão esquerdo empurra o material
    arrastar com o botão direito  atrai o material

Dependências: numpy e matplotlib  (pip install numpy matplotlib)
Para salvar .mp4 é preciso ter o ffmpeg instalado; .gif funciona com o pillow.
"""

import argparse
import time

import numpy as np

AGUA, GELEIA, NEVE = 0, 1, 2
NOMES = {AGUA: "água", GELEIA: "geleia", NEVE: "neve"}


def svd_2x2(F):
    """SVD analítica de matrizes 2x2 em lote: F = U diag(sig) Vᵀ.

    Muito mais rápida que np.linalg.svd para milhares de matrizes pequenas.
    1) decomposição polar F = R S (R rotação, S simétrica);
    2) diagonaliza S com uma rotação de Jacobi.
    """
    a00, a01, a10, a11 = F[:, 0, 0], F[:, 0, 1], F[:, 1, 0], F[:, 1, 1]
    x, y = a00 + a11, a10 - a01
    esc = 1.0 / np.sqrt(x * x + y * y)
    c, s = x * esc, y * esc                      # R = [[c, -s], [s, c]]
    S00 = c * a00 + s * a10
    S10 = -s * a00 + c * a10
    S11 = -s * a01 + c * a11
    tao = 0.5 * (S00 - S11)
    w = np.sqrt(tao * tao + S10 * S10)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(tao > 0, S10 / (tao + w), S10 / (tao - w))
    t = np.where(np.abs(S10) < 1e-6, 0.0, t)
    cj = 1.0 / np.sqrt(t * t + 1)
    sj = -t * cj
    s1 = cj * cj * S00 - 2 * cj * sj * S10 + sj * sj * S11
    s2 = sj * sj * S00 + 2 * cj * sj * S10 + cj * cj * S11
    troca = s1 < s2
    V = np.empty_like(F)
    V[:, 0, 0] = np.where(troca, -sj, cj); V[:, 0, 1] = np.where(troca, cj, sj)
    V[:, 1, 0] = np.where(troca, -cj, -sj); V[:, 1, 1] = np.where(troca, -sj, cj)
    sig = np.stack([np.where(troca, s2, s1), np.where(troca, s1, s2)], axis=1)
    sig = np.stack([np.maximum(sig[:, 0], sig[:, 1]), np.minimum(sig[:, 0], sig[:, 1])], axis=1)
    R = np.empty_like(F)
    R[:, 0, 0] = c; R[:, 0, 1] = -s; R[:, 1, 0] = s; R[:, 1, 1] = c
    U = R @ V
    return U, sig, np.transpose(V, (0, 2, 1))


class MLSMPM:
    """Simulador MLS-MPM 2D num domínio [0, 1] x [0, 1]."""

    def __init__(self, n_grade=64, dt=1e-4, E=5e3, nu=0.2):
        self.n = n_grade                    # células por eixo
        self.dx = 1.0 / n_grade
        self.inv_dx = float(n_grade)
        self.dt = dt
        self.p_vol = (self.dx * 0.5) ** 2   # 4 partículas por célula
        self.p_massa = self.p_vol * 1.0     # densidade 1
        self.gravidade = 50.0
        self.endurecimento = 10.0           # ξ da neve
        self.maciez_geleia = 0.3
        self.nu = nu
        self.definir_E(E)
        self.limpar()

    def definir_E(self, E):
        """Parâmetros de Lamé a partir do módulo de Young E e Poisson ν."""
        self.E = E
        self.mu0 = E / (2 * (1 + self.nu))
        self.lambda0 = E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

    # ------------------------------------------------------------------ #
    # Partículas
    # ------------------------------------------------------------------ #
    def limpar(self):
        self.x = np.zeros((0, 2))           # posição
        self.v = np.zeros((0, 2))           # velocidade
        self.F = np.zeros((0, 2, 2))        # gradiente de deformação
        self.C = np.zeros((0, 2, 2))        # matriz afim (APIC / MLS)
        self.Jp = np.zeros(0)               # razão de volume plástico (neve)
        self.material = np.zeros(0, dtype=np.int8)

    def _adicionar(self, pos, material, vel):
        k = len(pos)
        self.x = np.vstack([self.x, pos])
        self.v = np.vstack([self.v, np.tile(vel, (k, 1))])
        self.F = np.concatenate([self.F, np.tile(np.eye(2), (k, 1, 1))])
        self.C = np.concatenate([self.C, np.zeros((k, 2, 2))])
        self.Jp = np.concatenate([self.Jp, np.ones(k)])
        self.material = np.concatenate([self.material, np.full(k, material, np.int8)])

    def adicionar_retangulo(self, centro, tamanho, material, vel=(0.0, 0.0)):
        area = tamanho[0] * tamanho[1]
        k = int(area * self.n ** 2 * 4)
        pos = np.asarray(centro) + (np.random.rand(k, 2) - 0.5) * np.asarray(tamanho)
        self._adicionar(pos, material, vel)

    def adicionar_disco(self, centro, raio, material, vel=(0.0, 0.0)):
        k = int(np.pi * raio ** 2 * self.n ** 2 * 4)
        ang = np.random.rand(k) * 2 * np.pi
        r = raio * np.sqrt(np.random.rand(k))
        pos = np.asarray(centro) + np.stack([r * np.cos(ang), r * np.sin(ang)], 1)
        self._adicionar(pos, material, vel)

    def remover_perto(self, ponto, raio):
        manter = np.sum((self.x - ponto) ** 2, axis=1) > raio ** 2
        for nome in ("x", "v", "F", "C", "Jp", "material"):
            setattr(self, nome, getattr(self, nome)[manter])

    # ------------------------------------------------------------------ #
    # Um passo de tempo
    # ------------------------------------------------------------------ #
    def _pesos(self):
        """B-spline quadrática: cada partícula enxerga 3x3 nós da grade."""
        xg = self.x * self.inv_dx
        base = np.floor(xg - 0.5).astype(np.int64)          # nó do canto inferior esquerdo
        fx = xg - base                                       # posição relativa, em [0.5, 1.5)
        w = np.stack([0.5 * (1.5 - fx) ** 2,
                      0.75 - (fx - 1.0) ** 2,
                      0.5 * (fx - 0.5) ** 2], axis=1)        # (P, 3, 2)
        return base, fx, w

    def passo(self, ferramenta=None):
        """Avança a simulação em dt. Retorna o tempo (s) gasto em cada fase."""
        n, N, dx, dt = self.n, self.n + 1, self.dx, self.dt
        P = len(self.x)
        if P == 0:
            return 0.0, 0.0, 0.0
        t0 = time.perf_counter()

        # ================= 1. P2G (partícula -> grade) =================
        base, fx, w = self._pesos()

        # Atualiza F com o gradiente de velocidade. No MLS-MPM esse gradiente
        # é a própria matriz afim C da partícula:  F <- (I + dt C) F
        F = (np.eye(2) + dt * self.C) @ self.F

        # Modelo constitutivo: o que muda entre os materiais
        h = np.exp(self.endurecimento * (1.0 - self.Jp))     # endurecimento da neve
        h[self.material == GELEIA] = self.maciez_geleia
        mu = self.mu0 * h
        lam = self.lambda0 * h
        mu[self.material == AGUA] = 0.0                      # fluido: sem cisalhamento

        U, sig, Vt = svd_2x2(F)
        neve = self.material == NEVE
        if neve.any():
            # Plasticidade: limita os valores singulares; o excesso vira Jp
            sig_lim = np.clip(sig[neve], 1 - 2.5e-2, 1 + 4.5e-3)
            self.Jp[neve] *= np.prod(sig[neve] / sig_lim, axis=1)
            sig[neve] = sig_lim
            F[neve] = U[neve] @ (sig_lim[:, :, None] * Vt[neve])
        J = np.prod(sig, axis=1)
        agua = self.material == AGUA
        if agua.any():
            # Água guarda só a variação de volume
            F[agua] = np.sqrt(J[agua])[:, None, None] * np.eye(2)
        self.F = F

        # Tensão de Kirchhoff (corotacional fixo):
        #   τ = 2μ (F − R) Fᵀ + λ (J − 1) J I,   com R = U Vᵀ
        R = U @ Vt
        tau = (2 * mu)[:, None, None] * (F - R) @ np.transpose(F, (0, 2, 1))
        tau += (lam * (J - 1) * J)[:, None, None] * np.eye(2)

        # Contribuição central do MLS-MPM: a força interna e o momento afim
        # entram juntos numa única matriz por partícula.
        #   M⁻¹ = 4 / dx²  (para B-spline quadrática)
        termo_tensao = -dt * self.p_vol * 4 * self.inv_dx ** 2 * tau
        afim = termo_tensao + self.p_massa * self.C           # (P, 2, 2)

        # Espalha para os 3x3 nós vizinhos (os 9 nós de todas as partículas
        # de uma vez, e uma soma por índice com np.bincount)
        offs = np.array([(i, j) for i in range(3) for j in range(3)])          # (9, 2)
        peso = w[:, offs[:, 0], 0] * w[:, offs[:, 1], 1]                         # (P, 9)
        dpos = (offs[None, :, :] - fx[:, None, :]) * dx                          # (P, 9, 2)
        idx = (base[:, None, 0] + offs[:, 0]) + (base[:, None, 1] + offs[:, 1]) * N
        mv = self.p_massa * self.v
        contrib = peso[:, :, None] * (mv[:, None, :] + dpos @ np.transpose(afim, (0, 2, 1)))
        idx = idx.ravel()
        massa_grade = np.bincount(idx, (peso * self.p_massa).ravel(), N * N)
        momento_grade = np.stack([np.bincount(idx, contrib[:, :, 0].ravel(), N * N),
                                  np.bincount(idx, contrib[:, :, 1].ravel(), N * N)], axis=1)
        t1 = time.perf_counter()

        # ================= 2. Atualização da grade =================
        ativo = massa_grade > 0
        vel_grade = np.zeros_like(momento_grade)
        vel_grade[ativo] = momento_grade[ativo] / massa_grade[ativo, None]
        vel_grade[ativo, 1] -= dt * self.gravidade

        if not hasattr(self, "_ii") or len(self._ii) != N * N:
            ii, jj = np.meshgrid(np.arange(N), np.arange(N), indexing="xy")
            self._ii, self._jj = ii.ravel(), jj.ravel()      # índice = i + j*N
        ii, jj = self._ii, self._jj
        if ferramenta is not None:
            pos_no = np.stack([ii, jj], 1) * dx
            d = pos_no - ferramenta["pos"]
            dist = np.sqrt(np.sum(d ** 2, 1))
            dentro = ativo & (dist < ferramenta["raio"])
            k = (1 - dist[dentro] / ferramenta["raio"])[:, None]
            if ferramenta["modo"] == "empurrar":
                vel_grade[dentro] += (ferramenta["vel"] - vel_grade[dentro]) * 0.25 * k
            else:  # atrair
                vel_grade[dentro] -= d[dentro] / (dist[dentro, None] + 1e-6) * 0.6 * k

        # Paredes: zera a componente que entra na parede (3 células de margem)
        vx, vy = vel_grade[:, 0], vel_grade[:, 1]
        vx[(ii < 3) & (vx < 0)] = 0
        vx[(ii > n - 3) & (vx > 0)] = 0
        vy[(jj < 3) & (vy < 0)] = 0
        vy[(jj > n - 3) & (vy > 0)] = 0
        self.massa_grade = massa_grade
        t2 = time.perf_counter()

        # ================= 3. G2P (grade -> partícula) =================
        gv = vel_grade[idx].reshape(P, 9, 2)                                    # velocidade dos 9 nós
        nova_v = np.sum(peso[:, :, None] * gv, axis=1)
        dpos_g = dpos * self.inv_dx                                              # em unidades da grade
        novo_C = 4 * self.inv_dx * (np.transpose(peso[:, :, None] * gv, (0, 2, 1)) @ dpos_g)
        self.v = nova_v
        self.C = novo_C
        self.x = np.clip(self.x + dt * self.v, 1.5 * dx, 1 - 1.5 * dx)
        t3 = time.perf_counter()
        return t1 - t0, t2 - t1, t3 - t2


# ---------------------------------------------------------------------- #
# Cenas
# ---------------------------------------------------------------------- #
def cena_tres_materiais(s):
    s.adicionar_retangulo((0.30, 0.30), (0.2, 0.2), AGUA)
    s.adicionar_retangulo((0.55, 0.55), (0.2, 0.2), GELEIA)
    s.adicionar_retangulo((0.40, 0.80), (0.2, 0.2), NEVE)


def cena_represa(s):
    s.adicionar_retangulo((0.20, 0.30), (0.3, 0.5), AGUA)


def cena_geleias(s):
    s.adicionar_retangulo((0.30, 0.15), (0.16, 0.16), GELEIA)
    s.adicionar_retangulo((0.62, 0.15), (0.16, 0.16), GELEIA)
    s.adicionar_retangulo((0.46, 0.45), (0.20, 0.12), GELEIA)
    s.adicionar_disco((0.50, 0.80), 0.08, GELEIA, vel=(0, -2))


def cena_bola_de_neve(s):
    s.adicionar_retangulo((0.72, 0.17), (0.22, 0.26), NEVE)
    s.adicionar_disco((0.22, 0.24), 0.07, NEVE, vel=(7, 0.5))


def cena_geleia_na_agua(s):
    s.adicionar_retangulo((0.50, 0.12), (0.9, 0.18), AGUA)
    s.adicionar_disco((0.50, 0.65), 0.09, GELEIA, vel=(0, -1))


CENAS = [
    ("Três materiais", cena_tres_materiais),
    ("Represa", cena_represa),
    ("Geleias", cena_geleias),
    ("Bola de neve", cena_bola_de_neve),
    ("Geleia na água", cena_geleia_na_agua),
]


# ---------------------------------------------------------------------- #
# Visualização
# ---------------------------------------------------------------------- #
FUNDO = "#0d121b"


def cores_particulas(s):
    """Água clareia com a velocidade; neve compactada fica azulada."""
    c = np.zeros((len(s.x), 3))
    agua = s.material == AGUA
    vel = np.clip(np.linalg.norm(s.v[agua], axis=1) / 4, 0, 1)[:, None]
    c[agua] = np.array([0.24, 0.55, 1.0]) * (1 - vel) + np.array([0.75, 0.88, 1.0]) * vel
    c[s.material == GELEIA] = [1.0, 0.36, 0.45]
    neve = s.material == NEVE
    comp = np.clip((1 - s.Jp[neve]) * 4, 0, 1)[:, None]
    c[neve] = np.array([0.93, 0.95, 0.98]) * (1 - comp) + np.array([0.50, 0.68, 0.98]) * comp
    return c


def executar(args):
    import matplotlib
    if args.salvar:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    sim = MLSMPM(n_grade=args.grade)
    estado = {"cena": args.cena - 1, "pausado": False, "grade": args.grade_visivel,
              "mouse": None, "t": 0.0, "fases": np.zeros(3), "quadros": 0}

    def carregar(k):
        estado["cena"] = k
        sim.limpar()
        CENAS[k][1](sim)
        estado["t"] = 0.0

    carregar(estado["cena"])

    fig = plt.figure(figsize=(7, 7.4), facecolor=FUNDO)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.90])
    ax.set_facecolor("#070a10")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for borda in ax.spines.values():
        borda.set_color("#263044")
    N = sim.n + 1
    img_grade = ax.imshow(np.zeros((N, N)), extent=(-0.5 * sim.dx, 1 + 0.5 * sim.dx) * 2,
                          origin="lower", cmap="copper", vmin=0, vmax=sim.p_massa * 4,
                          alpha=0.55, interpolation="nearest", visible=False)
    pontos = ax.scatter([], [], s=args.tamanho_ponto, linewidths=0)
    titulo = fig.text(0.03, 0.955, "", color="#e7ecf3", fontsize=13, weight="bold")
    info = fig.text(0.03, 0.93, "", color="#8f9bb0", fontsize=9, family="monospace")

    def ao_clicar(ev):
        if ev.inaxes is ax and ev.button in (1, 3):
            estado["mouse"] = {"pos": np.array([ev.xdata, ev.ydata]), "ult": np.array([ev.xdata, ev.ydata]),
                               "modo": "empurrar" if ev.button == 1 else "atrair", "t": time.time()}

    def ao_mover(ev):
        m = estado["mouse"]
        if m is not None and ev.inaxes is ax:
            m["pos"] = np.array([ev.xdata, ev.ydata])

    def ao_soltar(ev):
        estado["mouse"] = None

    def ao_teclar(ev):
        if ev.key == " ":
            estado["pausado"] = not estado["pausado"]
        elif ev.key == "r":
            carregar(estado["cena"])
        elif ev.key == "g":
            estado["grade"] = not estado["grade"]
        elif ev.key in list("12345"):
            carregar(int(ev.key) - 1)

    fig.canvas.mpl_connect("button_press_event", ao_clicar)
    fig.canvas.mpl_connect("motion_notify_event", ao_mover)
    fig.canvas.mpl_connect("button_release_event", ao_soltar)
    fig.canvas.mpl_connect("key_press_event", ao_teclar)

    def atualizar(_):
        ferramenta = None
        m = estado["mouse"]
        if m is not None:
            agora = time.time()
            vel = (m["pos"] - m["ult"]) / max(agora - m["t"], 1e-3) * 0.12
            m["ult"], m["t"] = m["pos"].copy(), agora
            ferramenta = {"pos": m["pos"], "vel": vel, "raio": 0.08, "modo": m["modo"]}
        if not estado["pausado"]:
            for _ in range(args.subpassos):
                estado["fases"] += sim.passo(ferramenta)
            estado["t"] += args.subpassos * sim.dt
        estado["quadros"] += 1

        pontos.set_offsets(sim.x)
        pontos.set_color(cores_particulas(sim))
        img_grade.set_visible(estado["grade"])
        if estado["grade"] and hasattr(sim, "massa_grade"):
            img_grade.set_data(sim.massa_grade.reshape(N, N))
        f = estado["fases"] / max(estado["fases"].sum(), 1e-9) * 100
        titulo.set_text(f"MLS-MPM  ·  {CENAS[estado['cena']][0]}" + ("   [pausado]" if estado["pausado"] else ""))
        info.set_text(f"{len(sim.x)} partículas · grade {sim.n}×{sim.n} · t = {estado['t']:.2f} s · "
                      f"P2G {f[0]:.0f}%  grade {f[1]:.0f}%  G2P {f[2]:.0f}%")
        return pontos, img_grade, titulo, info

    if args.salvar:
        anim = FuncAnimation(fig, atualizar, frames=args.quadros, blit=False)
        writer = "pillow" if args.salvar.endswith(".gif") else "ffmpeg"
        inicio = time.time()
        anim.save(args.salvar, writer=writer, fps=30, dpi=args.dpi, savefig_kwargs={"facecolor": FUNDO})
        print(f"Salvo em {args.salvar} ({args.quadros} quadros, {time.time() - inicio:.0f} s)")
    else:
        anim = FuncAnimation(fig, atualizar, interval=1, blit=False, cache_frame_data=False)  # noqa: F841
        plt.show()


def main():
    p = argparse.ArgumentParser(description="Simulador MLS-MPM 2D (Hu et al., SIGGRAPH 2018)")
    p.add_argument("--cena", type=int, default=1, choices=range(1, len(CENAS) + 1),
                   help="cena inicial: 1 três materiais, 2 represa, 3 geleias, 4 bola de neve, 5 geleia na água")
    p.add_argument("--grade", type=int, default=64, help="resolução da grade (padrão 64; 96 ou 128 fica mais bonito e mais lento)")
    p.add_argument("--subpassos", type=int, default=15, help="subpassos de 1e-4 s por quadro desenhado")
    p.add_argument("--grade-visivel", action="store_true", help="começa mostrando a massa na grade")
    p.add_argument("--salvar", help="grava a animação em .gif ou .mp4 em vez de abrir a janela")
    p.add_argument("--quadros", type=int, default=240, help="quadros a gravar com --salvar")
    p.add_argument("--dpi", type=int, default=100)
    p.add_argument("--tamanho-ponto", type=float, default=1.5)
    executar(p.parse_args())


if __name__ == "__main__":
    main()
