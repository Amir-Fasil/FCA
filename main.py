from context import Context
from concept_lattice import ConceptLattice
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


dataframe2 = pd.read_csv("test_df.csv")
dataframe2.drop(columns=dataframe2.columns[0], inplace=True)  # Drop the first unnamed column
context2 = Context(dataframe2)
print("Dataframe2:\n", dataframe2)
print("Differentiation of {A, B}:", context2.Differentiate({"A", "B"}))
concepts2 = context2.extract_concepts()
print("Extracted Concepts:", concepts2.get_proper_concept())
print("Ordered_lattice", concepts2.get_lattice())