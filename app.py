import numpy as np
import pandas as pd
import panel as pn
import folium

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from bokeh.layouts import column, row, layout
from bokeh.models import (ColumnDataSource, BoxSelectTool, HoverTool, WheelZoomTool, Slider, Select, CustomJS)
from bokeh.palettes import d3, Spectral
from bokeh.plotting import figure, show, curdoc, output_file

from bokeh.plotting._stack import double_stack

# taken this from bokeh source code for varea_stack
def varea_step_stack(fig, stackers, **kw):
    result = []
    for kw in double_stack(stackers, "y1", "y2", **kw):
        result.append(fig.varea_step(**kw))
    return result

def load_xrf(filename):
    """Loads raw XRF data from Olympus Vanta, serial num 801256"""
    df = pd.read_excel(filename, sheet_name='Raw Data')
    
    # replace ND ppm and error with 0
    df = df.replace('ND', 0)
    df = df.fillna(0)

    # filter out elemental concentrations
    idxs = [i for i, n in enumerate(df.columns) if 'Concentration' in n]
    xrf = df.iloc[:,idxs]
    
    # delete "Concentration" from the column names
    xrf.columns = [name.split(' ')[0] for name in xrf.columns]
    
    # convert ppm to percentage 
    xrf /= 1e4
    
    # reorder columns based on decreasing mean concentration
    ordered_cols = []
    for idx in xrf.mean(axis=0).argsort():
        ordered_cols = [xrf.columns[idx]] + ordered_cols
    xrf = xrf[ordered_cols]
    
    # place depth as first column
    xrf.insert(0, 'DEPTH', df['Sample ID'])

    return xrf

def fit_pca(xrf):
    """Projects XRF data to 2-dimensions so that 
    the first dimension explains the most variance in readings
    the second dimension explains the second most variance, etc."""
    # project xrf points to principal components
    pca = PCA(n_components=3)
    xrf_proj = pca.fit_transform(xrf.iloc[:, 1:])
    xrf['PC1'] = xrf_proj[:,0]
    xrf['PC2'] = xrf_proj[:,1]
    xrf['PC3'] = xrf_proj[:,2]
    return xrf_proj, pca

# get file from google drive
url = 'https://docs.google.com/spreadsheets/d/1HweNrkLhSLam007IBMxkEWG1nB5cyYOi/edit?usp=share_link&ouid=102236031734985992074&rtpof=true&sd=true'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
# filename = 'data/Taparoo 78-3229-22S_Olympus_Vanta_Final.xlsx'

xrf = load_xrf(path)

# TODO: let user slice xrf, but for poc we'll do it manually here
# lateral starts around 10,000 meters
xrf = xrf.loc[xrf.DEPTH >= 10000]

# correlation heatmap
# corr_mat = xrf.iloc[:,1:].corr()
# corr_plot = sb.heatmap(corr_mat, 
#                        cmap=sb.diverging_palette(240, 40, as_cmap=True),
#                       xticklabels=True,
#                       yticklabels=True)
# corr_plot = pn.pane.Matplotlib(corr_plot.get_figure())

# project xrf points to principal components
pca = PCA(n_components=3)
xrf_proj = pca.fit_transform(xrf.iloc[:, 1:])
xrf['PC1'] = xrf_proj[:,0]
xrf['PC2'] = xrf_proj[:,1]
xrf['PC3'] = xrf_proj[:,2]

xrf['widths'] = np.ediff1d(xrf.DEPTH, to_begin=50)

def kmeans_fit(k):
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init='auto')
    kmeans.fit(xrf_proj)
    return kmeans

ks = [i for i in range(2, 10)]
kmeans = dict(zip(ks, [kmeans_fit(k) for k in ks]))
k_colors = d3['Category20'][max(ks)]
for k in ks:
    xrf[f'km{k}_colors'] = [k_colors[i] for i in kmeans[k].labels_]

xrf = ColumnDataSource(xrf)


#
# PCA CHART
#
ex_v = [str(x) for x in (pca.explained_variance_ratio_ * 100).round(2)]
pc = figure(title='Clustered XRF Readings', 
            height=300, 
            width=600, 
            tools='wheel_zoom,pan,lasso_select',
            x_axis_label=f'PC1 ({ex_v[0]}% variance)', 
            y_axis_label=f'PC2 ({ex_v[1]}% variance)',
            active_drag='lasso_select', active_scroll='wheel_zoom')
pc_km = pc.circle('PC1', 'PC2', source=xrf, 
                  color='km2_colors', 
                  selection_color='km2_colors', 
                  nonselection_alpha=0.2,
                  size=9, 
                  fill_alpha=0.5, 
                  line_color='km2_colors', 
                  name='pc_scatter')

def plot_centroids(k):
    x, y = kmeans[k].cluster_centers_[:,:2].T
    colors = k_colors[:k]
    return pc.square_pin(x, y, visible=False, size=15, fill_alpha=0.7, fill_color=colors, line_color='white')
    
centroids = [plot_centroids(k) for k in ks]
centroids[0].visible = True

# PC plot hover tool (todo: let user configure)
hover_tool = HoverTool(tooltips=[('depth', '@DEPTH'), 
                                 ('Ca', '@Ca'), 
                                 ('Si', '@Si'), 
                                 ('Al', '@Al'),
                                 ('Fe', '@Fe')])

pc.add_tools(hover_tool)

# now link k_slider value to glyph colors
k_slider = Slider(title='Number of bed clusters', 
                  start=min(ks), 
                  end=max(ks), 
                  step=1, 
                  value=2, 
                  align=('center'))
js_code = """
    var k = cb_obj.value;
    var kmc = `km${k}_colors`;
    renderer.glyph.fill_color = {field: kmc};
    renderer.glyph.line_color = {field: kmc};
    renderer.selection_glyph.fill_color = {field: kmc};
    renderer.selection_glyph.line_color = {field: kmc};
    for(var i=0; i<centroids.length; i++){
        centroids[i].visible = (ks[i] == k);
    }
"""
callback = CustomJS(args=dict(renderer=pc_km, centroids=centroids, ks=ks), code=js_code)
k_slider.js_on_change('value', callback)

# k_slider = pn.widgets.IntSlider(name='number of clusters', start=3, end=9, step=1, value=3)

#
# XRF CHARTS
#
all_maj_stacks = ['Si', 'Al', 'Fe', 'K', 'Ca', 'Mg', 'S', 'P', 'Ti', 'Mn', 'Ba', 'Sr'] 
maj_stacks = ['Ca', 'S', 'Ba', 'Ti', 'Mn', 'Sr']
maj_stacks_choice = pn.widgets.MultiChoice(name='Major Elements', 
                                           value=maj_stacks, 
                                           options=all_maj_stacks, 
                                           placeholder='click to add more')

mud_stacks = ['Si', 'Al', 'Fe', 'K', 'Mg', 'Zn', 'Zr']
mud_stacks_choice = pn.widgets.MultiChoice(name='Mud Elements', 
                                           value=mud_stacks, 
                                           options=mud_stacks, 
                                           placeholder='click to add more')

trc_stacks = ['Rb', 'Y', 'Nb', 'Se', 'U', 'Th', 'V', 'Mo']
trc_stacks_choice = pn.widgets.MultiChoice(name='Trace Metal Elements', 
                                           value=trc_stacks, 
                                           options=trc_stacks, 
                                           placeholder='click to add more')

def xrf_plot(stacks):
    colors = Spectral[11][:len(stacks)]
    p = figure(title='XRF readings', height=300, width=900, 
            x_range=(xrf.data['DEPTH'].min(), xrf.data['DEPTH'].max()), 
            x_axis_label='depth',
            y_axis_label='element %',
            tools='xpan,xwheel_zoom,reset',
            active_scroll='xwheel_zoom',
            active_drag=None)
    p.vbar_stack(stackers=stacks, x='DEPTH',
            width='widths',
            color=colors,
            alpha=0.8,
            legend_label=stacks,
            selection_color='red',
            source=xrf)
    p_varea = varea_step_stack(p, stackers=stacks, x='DEPTH',
            color=colors,
            alpha=0.4,
            legend_label=stacks,
            source=xrf,
            step_mode='center')
    p.legend.orientation = 'horizontal'

    # tools
    select_tool = BoxSelectTool(continuous=True, dimensions='width', persistent=True)
    select_tool.overlay.fill_color = 'grey'
    p.toolbar.active_drag = select_tool
    p.add_tools(select_tool)
    
    hover_tool = HoverTool(tooltips=[('element', '$name')], renderers=p_varea)
    p.add_tools(hover_tool)

    return p

maj = xrf_plot(maj_stacks)
mud = xrf_plot(mud_stacks)
trc = xrf_plot(trc_stacks)


######## NOW WORK IN PANEL
file_input = pn.widgets.FileInput()
maj_pane = pn.pane.Bokeh(pn.bind(xrf_plot, maj_stacks_choice))
mud_pane = pn.pane.Bokeh(pn.bind(xrf_plot, mud_stacks_choice))
trc_pane = pn.pane.Bokeh(pn.bind(xrf_plot, trc_stacks_choice))

accordion = pn.Accordion(('Major Elements', maj_pane), 
                         ('Mud Elements', mud_pane), 
                         ('Trace Metal Elements', trc_pane), 
                         active=[0,1,2])

# MAP
m = folium.Map(tiles='https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryTopo/MapServer/tile/{z}/{y}/{x}',
               attr='Tiles courtesy of the <a href="https://usgs.gov/">U.S. Geological Survey</a>',
               location=[40, -110], zoom_start=12)
folium_pane = pn.pane.plot.Folium(m, name='Well Location', height=300, width=300, align='center')
folium.Marker([40, -110], tooltip='Taparoo').add_to(m)
folium_pane.object = m

# TEMPLATE INTERFACE
template = pn.template.FastListTemplate(title='Taparoo Lateral XRF', 
                                        theme='dark',
                                        theme_toggle=True,
                                        accent='#00A170',
                                        shadow=True,
                                        sidebar_width=350, 
                                        collapsed_sidebar=True)

## SIDEBAR INTERFACE
# template.sidebar.append(file_input)
template.sidebar.append(k_slider)
template.sidebar.append(maj_stacks_choice)
template.sidebar.append(mud_stacks_choice)
template.sidebar.append(trc_stacks_choice)
# template.sidebar.append(pn.layout.Divider())
template.sidebar.append(folium_pane)

## MAIN INTERFACE
template.main.append(pn.Column(pc, accordion, sizing_mode='stretch_both'))
template.servable()
