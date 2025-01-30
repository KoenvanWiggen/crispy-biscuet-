#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 15:20:00 2024

@author: k.g.vanwiggen
"""

import numpy as np
import matplotlib.pyplot as plt

def create_box_plot(estimates, periods_to_plot):
    # Input
    method_names = ['VAR', 'LSE', 'MLE', 'Ridge', 'Alasso', 'Real']
    colors = ['black', 'grey', 'grey', 'black', 'black', 'black']
    line_styles = ['dotted', 'dashed', 'dashdot', 'dashed', 'dashdot', 'solid']
    num_methods = estimates.shape[2]

    # Make the plot
    fig, ax = plt.subplots(figsize=(12, 6))  
    
    group_width = 0.7
    method_width = group_width / num_methods
    positions = np.arange(periods_to_plot) + 1
    
    for t in range(periods_to_plot):
        for method in range(num_methods):
            pos = positions[t] + (method - (num_methods - 1) / 2) * method_width
            ax.boxplot(estimates[:, t, method], positions=[pos], widths=method_width,
                       boxprops=dict(linestyle=line_styles[method], linewidth=1.5, color = colors[method]),
                       whiskerprops=dict(linestyle=line_styles[method], linewidth=1.5, color = colors[method]),
                       capprops=dict(linestyle=line_styles[method], linewidth=1.5, color = colors[method]),
                       medianprops=dict(linestyle=line_styles[method], linewidth=1.5, color = colors[method]),
                       showfliers = False)

    # Adjusting the layout
    ax.set_title('Boxplots of the different methods', fontsize=16)
    ax.set_xlabel('Periods ahead', fontsize=14)
    ax.set_ylabel('gIRF value', fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.set_xticks([positions[t] for t in range(periods_to_plot)])
    ax.set_xticklabels([f'{i}' for i in range(periods_to_plot)], fontsize=10)
    legend_labels = [f'{i}' for i in method_names]
    legend_lines = [plt.Line2D([0], [0], color = colors[method] , linestyle=line_styles[method], linewidth=1.5) for method in range(num_methods)]
    ax.legend(legend_lines, legend_labels, loc='upper right', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12, length=8, width=1.5, pad=7)

    plt.tight_layout()
    plt.savefig('boxplot.png')
    plt.show()

    



