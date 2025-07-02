class ConceptLattice:

    def __init__(self, concepts: list):

        self.conceptLattice = concepts
        self.length = len(concepts)  # I can use this to quantify the number of node I need.

    def get_concept_lattice(self):
        """Returns the concept lattice."""
        return self.conceptLattice

    def _comp_variable(self, concept):

        """This function is used to return the comparing variables 
        using which the concepts are ordered. """

        return len(concept.get_extent())  #This is just temporary, It will change latter

    def get_ordered_lattice(self):
        
        """This function returns the ordered concept lattice."""
        
        # Sort the concepts based on the first element of the tuple (extent)
        ordered_concepts = sorted(self.conceptLattice, key=self._comp_variable)
        
        # Return the ordered list of concepts
        return ordered_concepts


