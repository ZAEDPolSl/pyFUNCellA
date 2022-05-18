import matplotlib.pyplot as plt

def move_legend(ax, new_loc, append_to_labels=None, **kws):
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    append_to_labels = [x for x in append_to_labels if x != 0]
    if append_to_labels is not None:
        labels = [labels[i]+": "+str(append_to_labels[i]) for i in range(len(labels))]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc=new_loc, title=title, **kws)