"""Reprodukovatelné české obrázky: pokusy, model a přípustná oblast."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

OUT = Path(__file__).resolve().parents[1] / 'obrazky'
BLUE, TEAL, RED = '#12355b', '#007f86', '#c73e1d'

def fuel(v):
    return 400 / np.asarray(v) + .05 * np.asarray(v) + 1.5

FLOW_WEIGHT = 1.0  # bezrozměrná váha odchylky průtoku; ukázka pro lambda = 1

def water(qt, qs, flow_weight=FLOW_WEIGHT):
    q = qt + qs
    t = (60 * qt + 10 * qs) / q
    return t, q, (t - 38.5)**2 + flow_weight * (q - 12)**2

def save(fig, name):
    fig.savefig(OUT / ('revize-' + name + '.svg'), bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 13, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'omm-revize-20260923'})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True, sharey=True)
    speeds = np.array([60., 120., 90.])
    for i, ax in enumerate(axes):
        ax.scatter(speeds[:i+1], fuel(speeds[:i+1]), s=75, color=TEAL, zorder=3)
        ax.set(xlim=(45, 135), ylim=(10, 12), xlabel='Rychlost (km/h)', title=f'{i+1}. pokus')
        ax.grid(alpha=.18)
        for v in speeds[:i+1]:
            ax.annotate(f'{fuel(v):.2f}'.replace('.', ','), (v, fuel(v)), xytext=(5, 9), textcoords='offset points')
    axes[0].set_ylabel('Spotřeba (l/100 km)')
    save(fig, 'auto-pokusy')
    v = np.linspace(45, 135, 400)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(v, fuel(v), color=TEAL, lw=3, label='Součet: 400/v + 0,05v + 1,5')
    ax.plot(v, 400/v, '--', color=BLUE, label='Klesající člen 400/v')
    ax.plot(v, .05*v+1.5, ':', color=RED, lw=2, label='Rostoucí člen 0,05v + 1,5')
    ax.scatter(speeds, fuel(speeds), color=TEAL, s=50, zorder=3)
    ax.set(xlabel='Rychlost (km/h)', ylabel='Spotřeba (l/100 km)', ylim=(0, 14))
    ax.legend(frameon=False, fontsize=11, loc='lower center', ncol=1)
    ax.grid(alpha=.18)
    save(fig, 'auto-model')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), sharey=True)
    for ax, upper in zip(axes, [130, 80]):
        ax.axvspan(45, 50, facecolor='#f8ded8', hatch='///', edgecolor=RED, alpha=.5)
        ax.axvspan(upper, 135, facecolor='#f8ded8', hatch='///', edgecolor=RED, alpha=.5)
        ax.plot(v, fuel(v), color=TEAL, lw=3)
        best = min(np.sqrt(8000), upper)
        ax.scatter([best], [fuel(best)], s=90, color=RED, zorder=4)
        ax.axvline(upper, color=RED, lw=1.5)
        ax.set(xlim=(45,135), ylim=(10.2,12.8), xlabel='Rychlost (km/h)', title=f'Povolená rychlost 50–{upper} km/h')
        ax.annotate(f'Optimum {best:.1f} km/h'.replace('.', ','), (best, fuel(best)), xytext=(-30,40), textcoords='offset points', arrowprops={'arrowstyle':'->'})
        ax.grid(alpha=.15)
    axes[1].text(105,12.1,'Nepřípustné', color=RED, ha='center')
    axes[0].set_ylabel('Spotřeba (l/100 km)')
    save(fig, 'auto-omezeni')
    qt, qs = np.meshgrid(np.linspace(.001,12,240), np.linspace(.001,12,240))
    t,q,f = water(qt, qs)
    fig, axes = plt.subplots(1, 2, figsize=(10.4,4))
    for ax, field, levels, title, target in zip(axes, [t,q], [np.arange(10,61,5),np.arange(0,25,2)], ['Teplota (°C)', 'Celkový průtok (l/min)'], [38.5,12]):
        c = ax.contour(qt,qs,field,levels=levels, colors=BLUE, linewidths=.7, alpha=.55)
        ax.clabel(c, inline=True, fontsize=9)
        ax.contour(qt,qs,field,levels=[target],colors=TEAL,linewidths=3)
        ax.scatter([6.84],[5.16],marker='*', s=160,color=RED,zorder=4)
        ax.set(xlabel='Teplá voda (l/min)', ylabel='Studená voda (l/min)', title=title, xlim=(0,12),ylim=(0,12))
    save(fig,'kohoutek-model')
    result = minimize_scalar(lambda cold: water(5,cold)[2],bounds=(0,12),method='bounded', options={'xatol':1e-12})
    assert abs(result.x-4.07873462)<1e-5
    fig, ax = plt.subplots(figsize=(8,4.4))
    c=ax.contour(qt,qs,f,levels=[1,5,10,20,40,80,160,320], colors=TEAL,linewidths=1)
    ax.clabel(c,inline=True,fontsize=10)
    ax.axvspan(5,12,facecolor='#f8ded8',hatch='///',edgecolor=RED,alpha=.45)
    ax.axvline(5,color=RED,lw=2)
    ax.scatter([6.84],[5.16],marker='x',s=100,color=BLUE,zorder=4)
    ax.annotate('Bez limitu: (6,84; 5,16)',(6.84,5.16),xytext=(7,8.5),arrowprops={'arrowstyle':'->'},fontsize=11)
    ax.scatter([5],[result.x],s=80,color=RED,zorder=5)
    ax.annotate('S limitem: (5; 4,08)',(5,result.x),xytext=(.3,1.6),arrowprops={'arrowstyle':'->'},fontsize=11)
    ax.text(8.5,11,'Nepřípustná oblast',ha='center',color=RED)
    ax.set(xlim=(0,12),ylim=(0,12),xlabel='Teplá voda (l/min)',ylabel='Studená voda (l/min)',title=f'Vrstevnice účelové funkce (λ = {FLOW_WEIGHT:g})')
    save(fig,'kohoutek-omezeni')
    measured_v = np.array([60., 120., 90.])
    design = np.column_stack([1/measured_v, measured_v, np.ones(3)])
    fitted = np.linalg.solve(design, fuel(measured_v))
    assert np.allclose(fitted, [400, .05, 1.5])
    for upper in [130, 80]:
        numeric = minimize_scalar(fuel, bounds=(50, upper), method='bounded')
        # A bounded optimizer approaches endpoints; include them explicitly.
        candidates = np.array([50., upper, numeric.x])
        best = candidates[np.argmin(fuel(candidates))]
        assert abs(best-min(np.sqrt(8000), upper)) < 1e-4
        print(f'Numericke optimum na [50,{upper}]: {best:.8f}')
    print(f'Parametry modelu odhadnute ze tri pokusu: {fitted}')
    print(f'Kohoutek: qt=5, qs={result.x:.8f}, T,Q,f={water(5,result.x)}')
    print(f'Auto: v={np.sqrt(8000):.8f}, f={fuel(np.sqrt(8000)):.8f}; s limitem80: {fuel(80)}')

if __name__ == '__main__':
    main()
