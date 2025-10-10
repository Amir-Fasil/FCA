from context import Context
from concept_lattice import ConceptLattice
import numpy as np
import pandas as pd

# data = {
#     "A" : [1, 1, 0],
#     "B" : [1, 0, 0],
#     "C" : [0, 1, 1]
# }

# dataframe = pd.DataFrame(data)
# context = Context(dataframe)
# print("Differentiation of {A, B}:", context.Differentiate({"A", "B"}))
# concepts = context.extract_concepts()
# print("Extracted Concepts:", concepts.get_proper_concept())
# print("Ordered_lattice", concepts.get_lattice())


# Name : A
# Cartoon : B
# Real : C
# Tortoise : D
# Dog : E
# Cat : F
# Mammal : G

# Garfield : 0
# Snoopy : 1
# Socks : 2
# Greyfriar’s Bobby : 3
# Harriet : 4


dataframe2 = pd.read_csv("test4.csv")
dataframe2.drop(columns=dataframe2.columns[0], inplace=True)  # Drop the first unnamed column
context2 = Context(dataframe2)
concepts2 = context2.extract_concepts()
concepts = concepts2.get_proper_concept()
print(f"number of concepts: {concepts}")
num_of_extent = [len(i[0]) for i in concepts]
print("Number of extents in each concept:", num_of_extent)
sorted_extents = np.argsort(num_of_extent)[::-1].tolist()
print("Sorted indices by extent size:", sorted_extents)
iceberg_concepts = [concepts[i] for i in sorted_extents[:32]]
print("Top 5 iceberg concepts (by extent size):", iceberg_concepts)
intents = [set(i[1]) for i in iceberg_concepts]
print("Intents of all concepts:", intents)
basis_attributes = concepts2.basis_attribute(context2.get_intents())
print("Basis attributes:", basis_attributes)
shared_intents = [len(intents[i].intersection(basis_attributes)) for i in range(len(intents))]
print("Number of basis attributes in each intent:", shared_intents)
total_shared = sum(shared_intents)
print("Total number of basis attributes in all intents:", total_shared)
#print("Extracted Concepts:", concepts2.get_concept_lattice())
print(concepts2.set_cover())
# concepts2.get_lattice()