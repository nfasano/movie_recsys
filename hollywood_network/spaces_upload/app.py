import gradio as gr
import pandas as pd
import numpy as np
from pyvis.network import Network

import pickle

with open("aa.txt", "rb") as f:
    aa = pickle.load(f)
with open("node_labels.txt", "rb") as f:
    node_labels = pickle.load(f)
with open("edges.txt", "rb") as f:
    edges = pickle.load(f)
df = pd.read_csv('df_out.csv', index_col=[0])


def needs_analysis():
    # compute size of nodes based on number of connections
    node_size = []
    for jactor in aa:
        actor_weight = df[(df['target'] == jactor) | (df['source'] == jactor)]['weight']
        node_size.append(np.sum(actor_weight))
    node_size = 50*np.array(node_size)/np.max(node_size)

    net = Network(notebook=True,cdn_resources='remote',bgcolor='#222222',font_color='white',height='750px',width='100%',select_menu=True,filter_menu=True)
    net.add_nodes(aa, label=node_labels[aa], size=node_size)
    # net.add_nodes(aa, size=node_size)
    net.add_edges(edges)
    net.repulsion(node_distance=200, spring_length=2)
    net.show_buttons()
    html = net.generate_html()
    #need to remove ' from HTML
    html = html.replace("'", "\"")
    
    return f"""<iframe style="width: 100%; height: 600px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""

with gr.Blocks() as demo:
    gr.HTML(value=needs_analysis())

# demo = gr.Interface(
#     needs_analysis,
#     inputs=None,
#     outputs=gr.outputs.HTML(),
#     title="pyvis_in_gradio",
#     allow_flagging='never'
# )

demo.launch()