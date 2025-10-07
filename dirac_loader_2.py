from context import Context
import pandas as pd
import numpy as np
from sympy import symbols, Poly
import qci_client as qc

client = qc.QciClient()


# Step 1: Load data

dataframe2 = pd.read_csv("test2.csv")
dataframe2.drop(columns=dataframe2.columns[0], inplace=True)

# Build FCA context
context2 = Context(dataframe2)
concepts2 = context2.extract_concepts()

# Step 2: Get multibody matrix

Q= concepts2.set_cover()

num_concepts = Q.shape[0]

# Step 3: Define binary variables (c1...cN)
c = symbols(f"c1:{num_concepts+1}")  # (c1, c2, ..., cN)

# Step 4: Build the multibody formulation
Obj_fun_expr = sum(Q[i, j] * c[i] * c[j] for i in range(num_concepts) for j in range(num_concepts))

# print("\nMUltibody formulation H(c) =")
# print(Obj_fun_expr)

# Step 5: Extract coefficients and indices (1-based indexing)
poly = Poly(Obj_fun_expr, *c)

poly_coefs = []
poly_indices = []

for i in range(num_concepts):
    for j in range(num_concepts):
        coeff = Q[i, j]
        if coeff != 0:
            if i == j:
                # Diagonal term → [0, i+1]
                idx = [0, i + 1]
            else:
                # Off-diagonal → [i+1, j+1] (always sorted ascending)
                idx = sorted([i + 1, j + 1])
            poly_indices.append(idx)
            poly_coefs.append(float(coeff))


print("\nPolynomial Coefficients:")
print(poly_coefs)
print(poly_coefs.__len__())
print("\nPolynomial Indices (1-based):")
print(poly_indices)


# Step 6: Encode for QCi upload

data = []
for i in range(len(poly_coefs)):
    data.append({
        "val": poly_coefs[i],
        "idx": poly_indices[i]
    })
poly_file = {"file_name": "test-polynomial",
             "file_config": {"polynomial": {
                 "min_degree": 1,
                 "max_degree": 2,
                 "num_variables":num_concepts ,
                 "data": data
             }}}
file_id = client.upload_file(file=poly_file)["file_id"]


job_body = client.build_job_body(
       job_type="sample-hamiltonian", 
       polynomial_file_id=file_id, 
       job_params={"device_type": "dirac-3", 
       "sum_constraint": num_concepts, "relaxation_schedule": 1, "num_samples":1 })

response = client.process_job(job_body=job_body)

print("Found solutions:")
print(response['results']['solutions'])
print("with energies:")
print(response['results']['energies'])
print("and counts:")
print(response['results']['counts'])