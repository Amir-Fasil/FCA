from context import Context
from concept_lattice import ConceptLattice
import pandas as pd

data = {
    "A" : [1, 1, 0],
    "B" : [1, 0, 0],
    "C" : [0, 1, 1]
}

dataframe = pd.DataFrame(data)
context = Context(dataframe)
print("Differentiation of {A, B}:", context.Differentiate({"A", "B"}))
concepts = context.extract_concepts()
print("Extracted Concepts:", concepts.get_concept_lattice())
print("Ordered_lattice", concepts.get_ordered_lattice())