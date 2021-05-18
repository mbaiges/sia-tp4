from ej1a import get_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import numpy as np

finished_avg_error = False
finished_hopfield = False

iteration = 0

def plot_avg_error(q):

    global finished_avg_error

    finished_avg_error = False

    iters_val = []
    # error
    avg_error = []

    fig = plt.figure()

    #creating a subplot 
    ax1 = fig.add_subplot(1,1,1)

    def animate(i): 
        global finished_avg_error, iteration

        if finished_avg_error:
            return

        avg_error_data = q.get()

        if avg_error_data == "STOP":
            finished_avg_error = True
            return

        iters_val.append(iteration)
        avg_error.append(avg_error_data['mean_error'])

        ax1.clear()

        plt.xlabel("Iterations")
        plt.ylabel("Avg Error")
        plt.title("Avg Error Real-Time")

        l1 = ax1.plot(iters_val, avg_error, 'r-')

        iteration += 1
        
    ani = animation.FuncAnimation(fig, animate, interval=5) 
    plt.show()

    return

def plot_hopfield(q):
    global finished_hopfield, iteration

    finished_hopfield = False

    fig, ax = plt.subplots()

    iteration = 0

    def animate(i): 
        global finished_hopfield, iteration

        if finished_hopfield:
            return

        hopfield_data = q.get()

        if hopfield_data == "STOP":
            finished_hopfield = True
            return

        mat = hopfield_data['output']

        ax.clear()
        ax.set_title(f"Hopfield outputs Real-Time (iteration={iteration})")
        im = plt.imshow(mat, cmap='Greys', interpolation='nearest')
        ax.set_xticks(np.arange(len(mat[0])))
        ax.set_yticks(np.arange(len(mat)))
        ax.set_xticklabels(range(len(mat[0])))
        ax.set_yticklabels(range(len(mat)))

        # Loop over data dimensions and create text annotations.

        plt.show()

        iteration += 1
        
    ani = animation.FuncAnimation(fig, animate, interval=500) 
    plt.show()

    return