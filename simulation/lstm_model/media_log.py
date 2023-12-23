
import wandb
import matplotlib.pyplot as plt

################################## WANDB FUNCTIONS FOR LOGGING VALUES TO THE SERVER ###################################

# title refers to image title represented in wandb
def image_log(image, title):
    wandb.log({title: [wandb.Image(image)]})

def graph_log(plot, title):
    wandb.log({title: plot})


############################################## PLOTTING GRAPH FUNCTIONS ###############################################

def plot_graph(ypred, ylab, config, title):
    """Interactive image"""
    fig, ax = plt.subplots()
    ax.plot(ypred, color='orange', label="Predicted")
    ax.plot(ylab, linestyle="dashed", linewidth=1, label="Actual")
    ax.grid(visible=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    # plt.xlim(left=0, right=306)
    ax.set_ylabel('Mean Air Temperature [°C]')
    ax.set_xlabel('Time [h]')
    ax.set_title(title)
    ax.legend()
    # plt.close()
    return fig

def plot_scatter(ypred, yreal):
    scatter = plt.figure()
    plt.scatter(yreal, ypred, edgecolor='white', linewidth=1, alpha=0.05)
    plt.grid(True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    # plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    plt.xlabel('Actual Temperature [°C]')
    plt.ylabel('Predicted Temperature [°C]')
    plt.title('Comparison among predicted and actual temperature \n in simulation environment using LSTM')
    # plt.show()
    # plt.close()
    return scatter

def error_distribution(ypred, yreal):
    import seaborn as sns
    from scipy.stats import gaussian_kde
    error = []
    for y_p, y_r in zip(ypred, yreal):
        error.append(y_p - y_r)

    fig, dens = plt.subplots()
    dens = sns.kdeplot(error, color="dodgerblue", lw=1, fill=True)

    # Here I get the vertices information for each axis
    p = dens.collections[0].get_paths()[0]
    v = p.vertices
    lx = [v[r][0] for r in range(len(v))]
    # ly = [v[r][1] for r in range(len(v))]

    # Then I plot the horizontal limits of lx
    # dens.axvline(min(lx)*0.75, color='r', linestyle='dashed')
    # dens.axvline(max(lx)*0.75, color='r', linestyle='dashed')
    dens.axvline(x=0, color='dodgerblue', linestyle='dashed')
    dens.set_xlim(-5, 5)
    dens.set_title('Error distribution of predicted Indoor Air temperature')
    # fig.show()
    # plt.close()
    return fig

def pth_temp_in_temp_out_plot(p_th, temp_in, temp_out, lab_pth, lab_temp_in, lab_temp_out, title):
    # add a new lineplot to the figure with temp_out
    p_th_rescaled = [value / 1 for value in p_th]  # Divide each value by 1000 to rescale the values

    fig, ax = plt.subplots()

    ax.plot(temp_in, label=lab_temp_in)
    ax.plot(temp_out, label=lab_temp_out, color='orange')
    ax.plot(p_th_rescaled, label=lab_pth, color='red')
    ax.set_xlabel('Time [h]')
    ax.set_ylabel(lab_temp_in + ' - ' + lab_temp_out + ' - ' + lab_pth)
    ax.legend()
    ax.set_title(title)

    ax.grid(visible=True, which='major', color='#666666', linestyle='-')
    ax.minorticks_on()

    plt.close()
    return fig

