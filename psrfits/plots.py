import numpy as np
import matplotlib.pyplot as plt

def spec_plot(ds, arr, cmap='viridis', colorbar=False):
    fig, ax = plt.subplots()
    arr_shifted = np.roll(arr, len(ds.phase)//2, axis=-1)
    phase_shifted = ds.phase - ds.phase[len(ds.phase)//2]
    pc = ax.pcolormesh(phase_shifted, ds.freq, arr_shifted, cmap=cmap)
    if colorbar: plt.colorbar(pc)
    ax.set(xlabel='Phase (ms)', ylabel='Frequency (MHz)')
    
    return ax
