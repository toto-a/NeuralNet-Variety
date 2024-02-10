import matplotlib.pyplot as plt
import numpy as np
from torch.optim import Adam,SGD
from matplotlib.animation import FuncAnimation
from optim_utils import rosenbrock,run_optimization
from optim_cstm import MyOptim


def create_path_visualizer(paths,colors,
                           names,
                           figsize=(12,12),
                           x_lim=(-2,2),
                           y_lim=(-1,3),
                           n_seconds=5):
    

    if not len(paths)==len(colors): 
        raise ValueError
    
    path_len=max(len(path) for path in paths)
    n_points=300
    x=np.linspace(*x_lim,n_points)
    y=np.linspace(*y_lim,n_points)
    X,Y=np.meshgrid(x,y)
    Z=rosenbrock(X,Y)

    global_min=(1.0,1.0)

    fig,ax=plt.subplots(figsize=figsize)
    ax.contour(X,Y,Z,90,cmap="jet")

    scatters=[ax.scatter(None,None,label=label,c=c) for c,label in zip(colors,names)]
    ax.legend(prop={"size":25})
    ax.plot(*global_min, "rD")


    def animate(i) :
        for path, scatter in zip(paths,scatters) :
            scatter.set_offsets(path[:i,:])
        
        ax.set_title(str(i))
    

    ms_per_frame=1000*n_seconds/path_len

    anim=FuncAnimation(fig,animate,frames=path_len,interval=ms_per_frame)

    return anim
    

if __name__=='__main__':
        x=0.3
        y=0.8
        n_iter=1500

        path_adam=run_optimization(Adam,x,y,n_iter)
        path_sgd=run_optimization(SGD,x,y,n_iter,lr=1e-3)
        path_cstm=run_optimization(MyOptim,x,y,n_iter,lr=1e-3)

        freq=10
        anim=create_path_visualizer([path_adam[::freq],path_sgd[::freq]],["r","b"],["Adam","SGD"],x_lim=(-1,1.1),y_lim=(-1,1.1),n_seconds=7)
        anim.save("./Custom_Optim/optim_comparison_all.gif")




