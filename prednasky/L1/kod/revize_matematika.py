"""Reprodukovatelné vektorové obrázky matematického dodatku přednášky 1."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1] / 'obrazky'
BLUE, TEAL, RED, GOLD = '#12355b', '#007f86', '#c73e1d', '#9a6200'

def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': BLUE, 'text.color': BLUE,
                         'svg.hashsalt': 'omm-l1-zaklady'})

def save(fig, name):
    fig.savefig(OUT / f'zaklady-{name}.svg', bbox_inches='tight', metadata={'Date': None})
    plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True)
    setup()
    # Jednotkové koule normy: hranice splňuje normu rovnou jedné.
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.5), layout='constrained')
    t = np.linspace(0, 2*np.pi, 500)
    curves = [(np.array([1,0,-1,0,1]), np.array([0,1,0,-1,0])),
              (np.cos(t), np.sin(t)),
              (np.array([1,1,-1,-1,1]), np.array([-1,1,1,-1,-1]))]
    for ax, (x,y), label in zip(axs, curves, ['$L_1$: součet velikostí', '$L_2$: běžná délka', '$L_\\infty$: největší složka']):
        ax.fill(x,y,color=TEAL,alpha=.12)
        ax.plot(x,y,color=TEAL,lw=3)
        ax.axhline(0,c=BLUE,lw=.7); ax.axvline(0,c=BLUE,lw=.7)
        ax.set(xlim=(-1.4,1.4),ylim=(-1.4,1.4),aspect='equal',title=label,xlabel='$x_1$',ylabel='$x_2$')
        ax.set_xticks([-1,0,1]); ax.set_yticks([-1,0,1])
    save(fig,'normy')
    fig, axs = plt.subplots(1,3,figsize=(11,3.4),layout='constrained')
    x=np.linspace(-1.7,1.7,201)
    for ax,y,title,col in zip(axs,[2*x,2*x+1,x*x],['Lineární: $2x$','Afinní: $2x+1$','Nelineární: $x^2$'],[TEAL,BLUE,RED]):
        ax.axhline(0,color='#aaaaaa',lw=.8); ax.axvline(0,color='#aaaaaa',lw=.8)
        ax.plot(x,y,color=col,lw=3); ax.scatter([0],[np.interp(0,x,y)],color=col,s=50,zorder=5)
        ax.set(xlabel='$x$',ylabel='$f(x)$',title=title,ylim=(-3.8,4.8))
    save(fig,'linearni')
    fig,ax=plt.subplots(figsize=(7,4),layout='constrained')
    x=np.linspace(-.2,3.1,250)
    ax.plot(x,(x-2)**2,color=BLUE,lw=3,label='$f(x)=(x-2)^2$')
    tangent_x=np.linspace(.05,1.95,50)
    ax.plot(tangent_x,1-2*(tangent_x-1),color=RED,lw=2,label='Tečna v bodě $x=1$')
    ax.scatter([1,2],[1,0],color=[RED,TEAL],s=60,zorder=4)
    ax.plot([1,1.5,1.5],[1,1,0],color=RED,ls='--',lw=1.5)
    ax.annotate('$\\Delta x=0{,}5$',(1.25,1),xytext=(1.3,1.5),ha='center')
    ax.annotate('$\\Delta f\\approx-1$',(1.5,.5),xytext=(1.8,.8))
    ax.set(xlabel='$x$',ylabel='$f(x)$',ylim=(-.35,5.1));ax.legend(loc='upper right')
    save(fig,'derivace')
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    a=np.linspace(-1.5,3,250); b=np.linspace(-1.6,2,250);xx,yy=np.meshgrid(a,b)
    contours=ax.contour(xx,yy,(xx-1)**2+2*yy**2,levels=[.5,1,2,3,5,8],colors=BLUE,alpha=.6)
    ax.clabel(contours,fontsize=10)
    ax.scatter([1],[0],color=TEAL,s=60)
    ax.scatter([0],[1],color=BLUE,s=50)
    ax.annotate('',xy=(-.6,2.2),xytext=(0,1),arrowprops={'arrowstyle':'->','color':RED,'lw':3})
    ax.annotate('',xy=(.5,0),xytext=(0,1),arrowprops={'arrowstyle':'->','color':TEAL,'lw':3})
    ax.text(-1.5,2.25,'Gradient: růst',color=RED)
    ax.text(.55,.45,'Krok proti gradientu',color=TEAL,fontsize=11)
    ax.text(1.1,-.25,'Minimum',color=TEAL)
    ax.set(xlabel='$x_1$',ylabel='$x_2$',ylim=(-1.5,2.6),aspect='equal')
    save(fig,'gradient')
    fig,axs=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    x=np.linspace(-1.5,1.5,200)
    axs[0].plot(x,x*x,lw=3,c=TEAL,label='$x^2$: miska')
    axs[0].plot(x,-x*x,lw=3,c=RED,label='$-x^2$: kopec')
    axs[0].plot(x,x**3,lw=2,c=BLUE,label='$x^3$: ani jedno')
    axs[0].scatter([0],[0],color=BLUE);axs[0].legend(fontsize=10);axs[0].set(xlabel='$x$',ylabel='$f(x)$',title='Ve všech případech $f\'(0)=0$')
    grid=np.linspace(-1.7,1.7,150);xx,yy=np.meshgrid(grid,grid)
    axs[1].contour(xx,yy,xx**2-yy**2,levels=[-2,-1,-.5,0,.5,1,2],cmap='coolwarm')
    axs[1].axhline(0,c=TEAL,lw=2);axs[1].axvline(0,c=RED,lw=2)
    axs[1].scatter([0],[0],color=BLUE,zorder=5)
    axs[1].set(xlabel='$x_1$: růst',ylabel='$x_2$: pokles',aspect='equal',title='Sedlo: $x_1^2-x_2^2$')
    save(fig,'krivost')
    fig,axs=plt.subplots(1,2,figsize=(10,3.3),layout='constrained')
    x=np.linspace(-2,2,200);u=np.linspace(.04,3,200)
    axs[0].plot(x,np.exp(x),c=TEAL,lw=3,label='$e^x$');axs[0].plot(u,np.log(u),c=BLUE,lw=3,label='$\\ln x$')
    axs[0].axhline(0,color='#aaaaaa',lw=.8);axs[0].axvline(0,color='#aaaaaa',lw=.8)
    axs[0].set(xlabel='$x$',ylabel='Hodnota',ylim=(-3.5,5));axs[0].legend()
    h=np.linspace(0,1,250);transport=h*np.exp(-2.5*h)
    axs[1].plot(h,transport,c=TEAL,lw=3);axs[1].scatter([.4],[.4*np.exp(-1)],c=RED,s=70,zorder=4)
    axs[1].axvline(.4,color=RED,ls='--',lw=1)
    axs[1].text(.48,.145,'Maximum při $H=0{,}4$',fontsize=11)
    axs[1].set(xlabel='Hematokrit $H$',ylabel='$D(H)=H e^{-2{,}5H}$',ylim=(0,.17))
    save(fig,'log-exp')
    # Číselné příklady použité na slidech, ověření bez solveru.
    A=np.array([[.5,.2],[3,2]]); x=np.array([100,150]);c=np.array([14,8])
    assert np.allclose(A@x,[80,600]) and c@x==2600
    p=np.array([0.,1.]); gradient=np.array([-2.,4.]);new=p-.25*gradient
    assert np.allclose(new,[.5,0]) and np.isclose((new[0]-1)**2+2*new[1]**2,.25)
    assert np.isclose(np.linalg.norm([3,4]),5)
    print('Vygenerováno 6 SVG; číselné příklady ověřeny.')

if __name__=='__main__':
    main()
