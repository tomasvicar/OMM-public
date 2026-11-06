"""Simulátor letu oštěpu (simulační úloha S11 – oštěp před rokem 1986 a po něm).

Oštěp je tuhá tyč letící ve svislé rovině. Stav je poloha těžiště (x, y),
jeho rychlost (vx, vy), úhel osy oštěpu nad vodorovnou theta a úhlová
rychlost omega. Na oštěp působí tíha a aerodynamická síla ze vzduchu,
který proti oštěpu fouká rychlostí u = v - (vitr, 0):

    nabeh alpha = theta - (smer vektoru u)
    q  = 1/2 * rho * |u|^2 * S
    vztlak  L = q * CL1 * sin(alpha) * cos(alpha)      (kolmo na u)
    odpor   D = q * (CD0 + CD2 * sin(alpha)^2)          (proti u)

Síla kolmá na osu N = L cos(alpha) + D sin(alpha) působí v působišti,
které leží d(alpha) = d0 + d1 |sin(alpha)| za těžištěm (záporné d =
před těžištěm). Moment kolem těžiště je M = -N * d(alpha), tedy
    m dv/dt = L + D + m g,   I d(omega)/dt = M.
Kladné d oštěp natáčí po proudu (hrotem k dráze letu), záporné d ho
naopak „zvedá“. Koeficienty jsou zjednodušené hladké křivky, nastavené
tak, aby model kvalitativně odpovídal pozorování (literatura je
v zadání úlohy S11, oddíl Literatura).

Oštěpy:
    "1984" – starý mužský oštěp: působiště při malém náběhu 4,5 cm PŘED
             těžištěm (d0 = -0,045 m),
    "1986" – nový oštěp: těžiště o 4 cm dopředu (d0 = -0,005 m).
Drsnost konce r v [0, 1] (zakázaná od 1991) posune působiště o 6r cm
dopředu (d0 -= 0,06 r) a přidá odpor (CD0 += 0,02 r).

Hod: těžiště startuje ve výšce 1,8 m nad x = 0 (odhodová čára).
Let končí, když těžiště klesne na zem (y = 0). Hod je PLATNÝ, jen když
oštěp dopadne hrotem napřed, tj. osa míří při dopadu dolů (theta < 0).

Rozhraní:
    hod(uhel_vyhozu, nabeh, uhlova_rychlost, ostep="1984", drsnost=0.0,
        rychlost=30.0, vitr=0.0, rtol=1e-7) -> dict
        uhel_vyhozu      směr rychlosti při výhozu nad vodorovnou [°]
        nabeh            úhel osy oštěpu vůči směru rychlosti při výhozu [°]
        uhlova_rychlost  počáteční otáčení oštěpu [°/s], kladné = hrot nahoru
        vitr             rychlost větru ve směru hodu [m/s] (záporná = protivítr)
      vrací: vysledek (m, oficiální výsledek = delka, nebo 0 pro neplatný hod),
             delka (m, místo dopadu bez ohledu na platnost), platny (bool),
             uhel_osy (° při dopadu, záporný = hrot dolů),
             uhel_drahy (° směr letu při dopadu), doba (s), max_vyska (m)
    trajektorie(...) – stejné argumenty, vrací (t, stav) pro vykreslení,
                       stav má řádky x, y, vx, vy, theta [rad], omega [rad/s]
    instance(seed)   – dict(rychlost, vitr) pro jiné podmínky hodu

Deterministický; jedno vyhodnocení trvá asi 10–30 ms.
"""

import math

import numpy as np
from scipy.integrate import solve_ivp

G = 9.81          # m/s^2
RHO = 1.2         # kg/m^3
VYSKA = 1.8       # výška těžiště při výhozu [m]

ZAKLAD = dict(m=0.8, I=0.40, S=0.0756, CL1=1.0, CD0=0.05, CD2=0.8, d1=0.10)
OSTEPY = {"1984": dict(ZAKLAD, d0=-0.045), "1986": dict(ZAKLAD, d0=-0.005)}


def parametry(ostep="1984", drsnost=0.0):
    """Parametry oštěpu včetně vlivu drsnosti konce."""
    p = dict(OSTEPY[ostep])
    r = float(np.clip(drsnost, 0.0, 1.0))
    p["d0"] -= 0.06 * r
    p["CD0"] += 0.02 * r
    return p


def _rovnice(p, vitr):
    m, I, k = p["m"], p["I"], 0.5 * RHO * p["S"]

    def f(t, s):
        x, y, vx, vy, th, om = s
        ux, uy = vx - vitr, vy                 # rychlost vůči vzduchu
        u = math.hypot(ux, uy)
        a = th - math.atan2(uy, ux)            # náběh
        sa, ca = math.sin(a), math.cos(a)
        q = k * u * u
        L = q * p["CL1"] * sa * ca
        D = q * (p["CD0"] + p["CD2"] * sa * sa)
        ex, ey = ux / u, uy / u
        N = L * ca + D * sa                    # síla kolmo na osu
        M = -N * (p["d0"] + p["d1"] * abs(sa))
        return [vx, vy, (-D * ex - L * ey) / m, (-D * ey + L * ex) / m - G, om, M / I]
    return f


def _let(uhel_vyhozu, nabeh, uhlova_rychlost, ostep, drsnost, rychlost, vitr,
         rtol, husta=False):
    p = parametry(ostep, drsnost)
    g0, a0, w0 = np.radians([uhel_vyhozu, nabeh, uhlova_rychlost])
    s0 = [0.0, VYSKA, rychlost * np.cos(g0), rychlost * np.sin(g0), g0 + a0, w0]

    def dopad(t, s):
        return s[1]
    dopad.terminal, dopad.direction = True, -1
    return solve_ivp(_rovnice(p, vitr), (0.0, 15.0), s0, events=dopad,
                     rtol=rtol, atol=rtol * 1e-2,
                     t_eval=np.linspace(0, 15, 3001) if husta else None)


def hod(uhel_vyhozu, nabeh, uhlova_rychlost, ostep="1984", drsnost=0.0,
        rychlost=30.0, vitr=0.0, rtol=1e-7):
    r = _let(uhel_vyhozu, nabeh, uhlova_rychlost, ostep, drsnost, rychlost,
             vitr, rtol)
    if len(r.t_events[0]) == 0:            # nedopadl do 15 s (nemělo by nastat)
        return dict(vysledek=0.0, delka=float("nan"), platny=False,
                    uhel_osy=float("nan"), uhel_drahy=float("nan"),
                    doba=float("nan"), max_vyska=float(r.y[1].max()))
    s = r.y_events[0][0]
    osa = float(np.degrees(s[4]))
    osa = (osa + 180.0) % 360.0 - 180.0
    platny = osa < 0.0
    delka = float(s[0])
    return dict(vysledek=delka if platny else 0.0, delka=delka, platny=platny,
                uhel_osy=osa, uhel_drahy=float(np.degrees(np.arctan2(s[3], s[2]))),
                doba=float(r.t_events[0][0]), max_vyska=float(r.y[1].max()))


def trajektorie(uhel_vyhozu, nabeh, uhlova_rychlost, ostep="1984", drsnost=0.0,
                rychlost=30.0, vitr=0.0, rtol=1e-7):
    r = _let(uhel_vyhozu, nabeh, uhlova_rychlost, ostep, drsnost, rychlost,
             vitr, rtol, husta=True)
    t, y = r.t, r.y
    if len(r.t_events[0]):
        t = np.append(t, r.t_events[0][0])
        y = np.hstack([y, r.y_events[0][0][:, None]])
    return t, y


def instance(seed):
    """Jiné podmínky hodu: rychlost výhozu 28–32 m/s, vítr -3 až +3 m/s."""
    rng = np.random.default_rng(seed)
    return dict(rychlost=round(float(rng.uniform(28.0, 32.0)), 2),
                vitr=round(float(rng.uniform(-3.0, 3.0)), 2))


if __name__ == "__main__":
    # ukázka volání s obyčejným výhozem (žádný doporučený hod)
    for o in ("1984", "1986"):
        print(o, hod(20.0, 5.0, -10.0, ostep=o))
