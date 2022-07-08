import numpy as np
import matplotlib.pyplot as plt

def plot_portrait(ds, ax, portrait, **kwargs):
    shifted_portrait = np.roll(portrait, len(ds.phase)//2, axis=-1)
    shifted_phase = ds.phase - ds.phase[len(ds.phase)//2]
    pc = ax.pcolormesh(shifted_phase, ds.freq, shifted_portrait, **kwargs)
    ax.set(xlabel='Phase (cycles)', ylabel='Frequency (MHz)')
    
    return pc
