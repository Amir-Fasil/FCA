import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict


class ConceptLattice:

    def __init__(self, concepts: list):

        self.conceptLattice = concepts
        self.length = len(concepts)  # I can use this to quantify the number of node I need.

    def get_concept_lattice(self):
        """Returns the concept lattice."""
        return self.conceptLattice

    def get_proper_concept(self):

        """This function retruns a list of concepts that are 
        tuples instead of Concept objects
        """
        proper_concepts = []
        for concept in self.conceptLattice:
            proper_concepts.append(concept.get_Concept())

        return proper_concepts
    
    def get_lattice(self):

        """This function returns the concept lattice."""


        concepts = self.get_proper_concept()
        concept_nodes = [(frozenset(extent), frozenset(intent)) for (extent, intent) in concepts]

        G = nx.Graph()
        G.add_nodes_from(concept_nodes)


        for i, (e1, i1) in enumerate(concept_nodes):
            for j, (e2, i2) in enumerate(concept_nodes):
                if i != j and i1 > i2 and len(i1) == len(i2) + 1:
                    G.add_edge((e1, i1), (e2, i2))


        levels = defaultdict(list)
        for concept in concept_nodes:
            level = len(concept[1]) 
            levels[level].append(concept)

 
        pos = {}
        for level, nodes_at_level in sorted(levels.items(), reverse=True):  
            spacing = 1.0
            width = (len(nodes_at_level) - 1) * spacing
            for i, node in enumerate(nodes_at_level):
                x = i * spacing - width / 2
                y = level
                pos[node] = (x, y)


        labels = {n: f"E:{set(n[0])}\nI:{set(n[1])}" for n in G.nodes()}
        nx.draw(G, pos, labels=labels, with_labels=True, node_color='lightcoral', node_size=1800, font_size=9)
        plt.title("Concept Lattice (Hasse Diagram, Ordered by Intent)")
        plt.gca().invert_yaxis()  
        plt.show()
        
        


