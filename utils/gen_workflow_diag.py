from graphviz import Digraph

def create_workflow_diagram():
    dot = Digraph(comment='Sentiment Pipeline Workflow', format='png')
    dot.attr(rankdir='LR', size='12,5')

    # Nodes: Data
    dot.node('Data', 'Input Text\n(CSV Data)', shape='cylinder', style='filled', color='lightgrey')
    dot.node('Stage1', 'Stage 1\nTranslation Decision', shape='diamond', style='filled', color='lightblue')
    dot.node('Stage2', 'Stage 2\nModel Selection', shape='diamond', style='filled', color='lightblue')
    dot.node('Output', 'Final Prediction\n(Sentiment)', shape='ellipse', style='filled', color='lightgreen')

    # Nodes: Agent
    dot.node('Agent', 'Context-Aware\nQ-Learning Agent', shape='box', style='filled', color='purple', fontcolor='white')
    
    # Nodes: Rewards
    dot.node('Cost', 'Cost\n(Latency/API)', shape='octagon', style='filled', color='salmon')
    dot.node('Fairness', 'Fairness\n(WRS Penalty)', shape='octagon', style='filled', color='salmon')
    
    # Edges: Flow
    dot.edge('Data', 'Stage1', label='Current State')
    dot.edge('Stage1', 'Stage2', label='Action 1\n(e.g. trans_none)')
    dot.edge('Stage2', 'Output', label='Action 2\n(e.g. m_textblob)')
    
    # Edges: Agent Interaction
    dot.edge('Data', 'Agent', label='State\nObservation', style='dashed')
    dot.edge('Agent', 'Stage1', label='Select Action', style='dashed')
    dot.edge('Agent', 'Stage2', label='Select Action', style='dashed')
    
    # Edges: Rewards
    dot.edge('Stage1', 'Cost', label='+Cost')
    dot.edge('Stage2', 'Cost', label='+Cost')
    dot.edge('Output', 'Fairness', label='Bias Check')
    
    dot.edge('Cost', 'Agent', label='Negative Reward', style='dotted')
    dot.edge('Fairness', 'Agent', label='Negative Reward', style='dotted')

    # Top Pipeline Box
    with dot.subgraph(name='cluster_results') as c:
        c.attr(style='filled', color='lightyellow')
        c.node_attr.update(style='filled', color='white')
        c.label = 'Top Converged Pipelines (Results)'
        c.node('P1', '1. trans_none -> m_textblob (84%)')
        c.node('P2', '2. trans_spanish -> m_textblob (6%)')
        c.node('P3', '3. trans_none -> m_random (4%)')

    dot.edge('Output', 'P1', style='invis')

    output_path = 'workflow_diagram'
    dot.render(output_path, view=True)
    print(f"Diagram saved to {output_path}.png")

if __name__ == "__main__":
    try:
        create_workflow_diagram()
    except ImportError:
        print("Graphviz not installed. Please run: pip install graphviz")