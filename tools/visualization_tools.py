from matplotlib.pyplot import *
import matplotlib.pyplot as plt
import numpy as np

def matrix_list_multiplication(matrix_A=None, list_B=None):
    if list_B is None:
        list_B = []
    AB = []
    for t in range(len(list_B)):
        AB.append(np.dot(matrix_A,list_B[t]))
    return AB

def keep_showing_figures():
    show()

def plot_heat_map(x=None, Bu=None, myTitle='title', outputFileName=None, left_title=None, right_title=None):

    figure()
    suptitle(myTitle)

    logmin = -4
    logmax = 0

    plt_x  = np.asarray(np.concatenate(x, axis=1))
    plt_Bu = np.asarray(np.concatenate(Bu,axis=1))

    np.seterr(divide = 'ignore') 
    plt_x  = np.log10(np.absolute(plt_x))
    plt_Bu = np.log10(np.absolute(plt_Bu))
    np.seterr(divide = 'warn') 

    # cut at the min
    plt_x  = np.clip(plt_x,  logmin - 1, logmax + 1)
    plt_Bu = np.clip(plt_Bu, logmin - 1, logmax + 1)

    if Bu is None:  # or pure zero?
        # don't subplot; plot only x
        pcolor(
            plt_x,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        colorbar()
        if left_title is not None:
            title(left_title)
        else:
            title('log10(|x|)')
        xlabel('Time')
        ylabel('Space')
        
    else:
        subplot(1,2,1)
        pcolor(
            plt_x,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        colorbar()
        if left_title is not None:
            title(left_title)
        else:
            title('log10(|x|)')
        xlabel('Time')
        ylabel('Space')

        subplot(1,2,2)
        pcolor(
            plt_Bu,
            cmap='jet',
            vmin=logmin,
            vmax=logmax
        )
        colorbar()
        if right_title is not None:
            title(right_title)
        else:
            title('log10(|u|)')
        xlabel('Time')

    if outputFileName is not None:
        # output as csv file
        np.savetxt(outputFileName+'-x.csv', plt_x.round(2).T, fmt='%.2f')
        np.savetxt(outputFileName+'-Bu.csv', plt_Bu.round(2).T, fmt='%.2f')

    show(block=False)

def plot_line_chart(list_x=None, list_y=None, title='title', xlabel='xlabel', ylabel='ylabel', line_format='o-', invert_x=False):
    if list_x is None:
        list_x = []
    if list_y is None:
        list_y = []
    figure()
    plot(list_x,list_y,line_format)
    suptitle(title)
    if invert_x:
        gca().invert_xaxis()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    show(block=False)

def plot_time_trajectory(x=None, Bu=None, xDes=None):
    '''
    Plots time trajectories and errors (useful for trajectory tracking)
    Inputs
       x, Bu : state and actuation values at nodes
       xDes  : desired trajectory
    '''
    figure()

    # nothing to plot
    if x is None:
        return

    TMax = len(x)
    if TMax < 1:
        return

    n_x = x[0].shape[0]

    def get_max_from_list (value, list_x):
        for tmp in list_x:
            val = np.max(tmp)
            if value < val:
                value = val

    def get_min_from_list (value, list_x):
        for tmp in list_x:
            val = np.min(tmp)
            if value > val:
                value = val

    #maxy = max([max(vec(x)) max(vec(Bu)) max(vec(xDes))]) + 2;
    maxy = np.max(x[0])
    get_max_from_list (maxy, x)
    get_max_from_list (maxy, Bu)
    get_max_from_list (maxy, xDes)
    maxy += 0.5

    #miny = min([min(vec(x)) min(vec(Bu)) max(vec(xDes))]) - 2;
    miny = np.min(x[0])
    get_min_from_list (miny, x)
    get_min_from_list (miny, Bu)
    get_min_from_list (miny, xDes)
    miny -= 0.5

    err = []
    maxe = None
    mine = None
    for i in range(len(xDes)):
        if i < len(x):
            val = np.absolute(xDes[i]-x[i])
            maxv = np.max(val)
            if maxe is None:
                maxe = maxv
            elif maxe < maxv:
                maxe = maxv
            minv = np.min(val)
            if mine is None:
                mine = minv
            elif mine > minv:
                mine = minv
            err.append(val)
    err = np.array(err)

    TMax_series = np.arange(1,TMax + 1)

    for node in range(n_x):
        # subplot(n_x, 2, node * 2 + 1)
        subplot(n_x, 1, node + 1)
        step(TMax_series, xDes[:, node])
        step(TMax_series, Bu[:, node])
        xticks([])
        ylabel('%d' % (node+1))
        ylim((miny,maxy))

        # subplot(n_x, 2, node * 2 + 2)
        # step(TMax_series, x[:, node])
        # xticks([])
        # ylabel('%d' % (node+1))
        # if mine != maxe:
        #     ylim((mine,maxe))

    # subplot(n_x, 2, n_x * 2 - 1)
    subplot(n_x, 1, n_x)
    legend(['disturbance', 'Bu'])
    xlabel('time step')

    # subplot(n_x, 2, n_x * 2)
    # legend('x')
    # xlabel('time step')

    show(block=False)
