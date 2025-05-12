import graphviz
import os
from settings.get_settings import join_strings, SETTINGS

#make vertical flowchart so it fits in column
#get rid of data selection container
#
def create_flowchart():
    """Create a professional flowchart of the VIMS image processing pipeline"""
    
    # Create a new directed graph
    dot = graphviz.Digraph(comment='VIMS Image Processing Pipeline')
    
    # Set graph attributes for a professional look
    dot.attr(rankdir='TB',  # Top to bottom layout
            splines='polyline',  # Cleaner straight lines with slight bends
            nodesep='0.25',  # Increased node separation
            ranksep='0.6',   # Increased rank separation
            fontname='Arial',
            bgcolor='white',
            compound='true',
            pad='0.5')  # Add padding around the entire graph
    
    # Layout control - create invisible edges to force layout
    dot.attr('edge', style='invis')
    
    # Create main layout subgraph to control overall structure
    with dot.subgraph(name='cluster_main_layout') as main:
        main.attr(rank='same', style='invis')
        main.node('top_anchor', '', style='invis', shape='point')
        main.node('middle_anchor', '', style='invis', shape='point')
        main.node('bottom_anchor', '', style='invis', shape='point')
        main.edge('top_anchor', 'middle_anchor')
        main.edge('middle_anchor', 'bottom_anchor')

    # Reset edge style for actual connections
    dot.attr('edge', style='')
    
    # Define different node styles with updated professional colors
    styles = {
        'data': {
            'shape': 'cylinder',
            'style': 'filled',
            'fillcolor': '#E3F2FD',  # Light blue
            'color': '#1976D2',  # Darker blue
            'fontname': 'Arial',
            'fontsize': '10',  # Slightly smaller font
            'height': '0.4',
            'width': '1.0',
            'penwidth': '1.0'
        },
        'process': {
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': '#FAFAFA',  # Very light gray
            'color': '#424242',  # Dark gray
            'fontname': 'Arial',
            'fontsize': '10',
            'height': '0.3',
            'width': '0.8',
            'penwidth': '1.0'
        },
        'decision': {
            'shape': 'diamond',
            'style': 'filled',
            'fillcolor': '#FFF3E0',  # Light orange
            'color': '#F57C00',  # Dark orange
            'fontname': 'Arial',
            'fontsize': '10',
            'height': '0.4',
            'width': '0.8',
            'penwidth': '1.0'
        },
        'subgraph': {
            'style': 'rounded',
            'color': '#9E9E9E',  # Medium gray
            'fontname': 'Arial Bold',
            'fontsize': '11',
            'margin': '12',  # Increased margin
            'penwidth': '0.8'
        },
        'iteration': {
            'shape': 'box',
            'style': 'rounded,filled,dashed',
            'fillcolor': '#F5F5F5',  # Light gray
            'color': '#616161',  # Darker gray
            'fontname': 'Arial',
            'fontsize': '10',
            'height': '0.3',
            'width': '0.8',
            'penwidth': '1.0'
        },
        'special': {
            'shape': 'box',
            'style': 'rounded,filled',
            'fillcolor': '#E8F5E9',  # Light green
            'color': '#2E7D32',  # Dark green
            'fontname': 'Arial',
            'fontsize': '10',
            'height': '0.3',
            'width': '0.8',
            'penwidth': '1.0'
        }
    }

    # Update edge style for a cleaner look
    edge_style = {
        'dir': 'forward',
        'penwidth': '0.8',
        'fontsize': '9',  # Smaller font for edge labels
        'color': '#757575',  # Lighter gray
        'arrowsize': '0.6',  # Smaller arrows
        'arrowhead': 'vee'
    }

    # Create subgraphs for different sections - now arranged vertically
    with dot.subgraph(name='cluster_input') as s:
        s.attr(label='Data Sources', **styles['subgraph'])
        s.attr(rank='same')
        s.node('cassini_v', 'VIMS-V\n96 bands', **styles['data'])  # Shortened labels
        s.node('cassini_ir', 'VIMS-IR\n256 bands', **styles['data'])
        s.node('srtc', 'SRTC++', **styles['data'])

    # VIMS-V Preprocessing subgraph
    with dot.subgraph(name='cluster_preprocess_v') as s:
        s.attr(label='VIMS-V Preprocessing', **styles['subgraph'])
        s.attr(rank='same')
        s.node('sky_baseline', 'Sky Pixel\nBaseline', **styles['process'])
        s.node('vert_correct', 'Vertical\nCorrection', **styles['process'])
        s.node('noise_reduce', 'Noise\nReduction', **styles['process'])
        s.node('validate', 'Quality\nCheck', **styles['decision'])

    with dot.subgraph(name='cluster_processing') as s:
        s.attr(label='Core Processing', **styles['subgraph'])
        s.node('ground', 'Ground\nDetection', **styles['special'])
        s.node('limb', 'Limb\nIsolation', **styles['process'])
        s.node('band_filter', 'Band\nFiltering', **styles['process'])
        s.node('center', 'Center\nDetection', **styles['process'])
        s.node('upsample', '5x\nUpsampling', **styles['process'])
        s.node('emission', 'Emission\nAngles', **styles['process'])

    with dot.subgraph(name='cluster_transects') as s:
        s.attr(label='Transect Analysis', **styles['subgraph'])
        s.node('north_sample', '30°N', **styles['special'])  # Shortened labels
        s.node('south_sample', '30°S', **styles['special'])
        s.node('orientation', 'E/W\nOrientation', **styles['process'])
        s.node('geometry', 'Viewing\nGeometry', **styles['process'])

    with dot.subgraph(name='cluster_analysis') as s:
        s.attr(label='Scientific Analysis', **styles['subgraph'])
        s.node('lm_fit', 'L-M\nFitting', **styles['special'])
        s.node('coeff_extract', 'Coefficient\nExtraction', **styles['process'])
        s.node('contrast', 'Brightness', **styles['process'])  # Shortened labels
        s.node('boundary', 'Boundary\nDetection', **styles['process'])
        s.node('hemispheric', 'Hemispheric\nAnalysis', **styles['process'])
        s.node('iteration', 'Parameter\nRefinement', **styles['iteration'])

    with dot.subgraph(name='cluster_output') as s:
        s.attr(label='Quality Control', **styles['subgraph'])  # Shortened label
        s.node('snr', 'SNR', **styles['decision'])  # Shortened labels
        s.node('stats', 'Validation', **styles['decision'])
        s.node('output', 'Products', **styles['data'])

    with dot.subgraph(name='cluster_global') as s:
        s.attr(label='Global Analysis', **styles['subgraph'])  # Shortened label
        s.node('data_agg', 'Data\nAggregation', **styles['special'])
        s.node('trend_analysis', 'Trend\nAnalysis', **styles['process'])
        s.node('dist_plots', 'Distribution', **styles['process'])  # Shortened labels
        s.node('wave_analysis', 'Wavelength', **styles['process'])
        s.node('trust_maps', 'Trust\nMapping', **styles['process'])
        s.node('global_viz', 'Visualization', **styles['data'])  # Shortened labels
        s.node('publication', 'Publication', **styles['data'])

    # Add edges with meaningful labels and better routing
    edge_style = {'dir': 'forward', 'penwidth': '0.8', 'fontsize': '8'}  # Thinner lines, smaller font
    
    # Data source routing
    dot.edge('cassini_v', 'sky_baseline', **edge_style)
    dot.edge('cassini_ir', 'ground', 'IR', **edge_style)  # Shortened label
    
    # VIMS-V preprocessing
    dot.edge('sky_baseline', 'vert_correct', **edge_style)
    dot.edge('vert_correct', 'noise_reduce', **edge_style)
    dot.edge('noise_reduce', 'validate', **edge_style)
    dot.edge('validate', 'ground', 'Pass', **edge_style)
    dot.edge('validate', 'sky_baseline', 'Fail', **edge_style)
    
    # Core processing
    dot.edge('srtc', 'ground', **edge_style)
    dot.edge('ground', 'limb', **edge_style)
    dot.edge('limb', 'band_filter', **edge_style)
    dot.edge('band_filter', 'center', **edge_style)
    dot.edge('center', 'upsample', **edge_style)
    dot.edge('upsample', 'emission', **edge_style)
    
    # Transect analysis
    dot.edge('emission', 'north_sample', **edge_style)
    dot.edge('emission', 'south_sample', **edge_style)
    dot.edge('north_sample', 'orientation', **edge_style)
    dot.edge('south_sample', 'orientation', **edge_style)
    dot.edge('orientation', 'geometry', **edge_style)
    
    # Scientific analysis
    dot.edge('geometry', 'lm_fit', **edge_style)
    dot.edge('lm_fit', 'coeff_extract', **edge_style)
    dot.edge('coeff_extract', 'contrast', **edge_style)
    dot.edge('contrast', 'boundary', **edge_style)
    dot.edge('boundary', 'hemispheric', **edge_style)
    dot.edge('hemispheric', 'iteration', **edge_style)
    dot.edge('iteration', 'lm_fit', 'Refine', **edge_style)
    dot.edge('iteration', 'snr', 'Done', **edge_style)  # Shortened label
    
    # Quality control and output
    dot.edge('snr', 'stats', 'Pass', **edge_style)
    dot.edge('stats', 'output', 'Pass', **edge_style)
    dot.edge('snr', 'ground', 'Fail', **edge_style)
    dot.edge('stats', 'ground', 'Fail', **edge_style)

    # Global analysis connections
    dot.edge('output', 'data_agg', **edge_style)
    dot.edge('data_agg', 'trend_analysis', **edge_style)
    dot.edge('data_agg', 'dist_plots', **edge_style)
    dot.edge('trend_analysis', 'wave_analysis', **edge_style)
    dot.edge('trend_analysis', 'trust_maps', **edge_style)
    dot.edge('wave_analysis', 'global_viz', **edge_style)
    dot.edge('trust_maps', 'global_viz', **edge_style)
    dot.edge('dist_plots', 'global_viz', **edge_style)
    dot.edge('global_viz', 'publication', **edge_style)

    # Save the flowchart with adjusted size for column width
    save_path = join_strings(SETTINGS["paths"]["parent_figures_path"], 
                           SETTINGS["paths"]["dev_figures_sub_path"],
                           "holistic_stats",
                           "vims_processing_flowchart")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set graph size and DPI for column width
    dot.attr(size='3.5,14')  # Slightly taller to accommodate better spacing
    dot.attr(dpi='300')
    
    # Render in multiple formats
    dot.render(save_path, format='svg', cleanup=True)
    dot.render(save_path + '_highres', format='png', cleanup=True)
    
    return {
        'svg': save_path + '.svg',
        'png': save_path + '_highres.png'
    }

if __name__ == "__main__":
    create_flowchart() 