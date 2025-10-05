from context import Context
import pandas as pd
import numpy as np
from sympy import symbols, Poly
import qci_client as qc

client = qc.QciClient()

# ----------------------------
# Step 1: Load data
# ----------------------------
dataframe2 = pd.read_csv("test_df.csv")
dataframe2.drop(columns=dataframe2.columns[0], inplace=True)

# Build FCA context
context2 = Context(dataframe2)
concepts2 = context2.extract_concepts()

# ----------------------------
# Step 2: Get QUBO matrix
# ----------------------------
Q= concepts2.set_cover()

num_concepts = Q.shape[0]

# print("QUBO Matrix Shape:", Q.shape)
print(Q)
# print("Number of concepts:", num_concepts)

# ----------------------------
# Step 3: Define binary variables (c1...cN)
# ----------------------------
c = symbols(f"c1:{num_concepts+1}")  # (c1, c2, ..., cN)

# ----------------------------
# Step 4: Build the QUBO formulation
# ----------------------------
qubo_expr = sum(Q[i, j] * c[i] * c[j] for i in range(num_concepts) for j in range(num_concepts))

# print("\nQUBO formulation H(c) =")
# print(qubo_expr)

# ----------------------------
# Step 5: Extract coefficients and indices (1-based indexing)
# ----------------------------
poly = Poly(qubo_expr, *c)

poly_coefs = []
poly_indices = []

for monom, coeff in poly.terms():
    poly_coefs.append(float(coeff))
    term_indices = []
    # 1-based indexing
    for var_index, exp in enumerate(monom, start=1):
        term_indices.extend([var_index] * exp)
    poly_indices.append(term_indices)

print("\nPolynomial Coefficients:")
print(poly_coefs)
print(poly_coefs.__len__())
print("\nPolynomial Indices (1-based):")
print(poly_indices)

# ----------------------------
# Step 6: Encode for QCi upload
# ----------------------------
data_int_problem = [{"idx": list(idx), "val": val} for idx, val in zip(poly_indices, poly_coefs)]

file_int_problem = {
    "file_name": "dirac_qubo_example",
    "file_config": {
        "polynomial": {
            "num_variables": num_concepts,   # number of variables (concepts)
            "min_degree": 2,                 
            "max_degree": 2,
            "data": data_int_problem,
        }
    }
}

file_response_int_problem = client.upload_file(file=file_int_problem)

# ----------------------------
# Step 7: Build and run job
# ----------------------------
job_body_int_problem = client.build_job_body(
    job_type='sample-hamiltonian-integer',
    job_name='test_qubo_job',
    job_params={
        'device_type': 'dirac-3',
        'num_samples': 1,                   # if request multiple samples make more that 1 
        'relaxation_schedule': 1,
        'num_levels': [2] * num_concepts
                # binary vars
    },
    polynomial_file_id=file_response_int_problem['file_id'],
)

job_response_int_problem = client.process_job(job_body=job_body_int_problem)

# ----------------------------
# Step 8: Inspect results
# ----------------------------
assert job_response_int_problem["status"] == qc.JobStatus.COMPLETED.value

print("Found solutions:")
print(job_response_int_problem['results']['solutions'])
print("with energies:")
print(job_response_int_problem['results']['energies'])
print("and counts:")
print(job_response_int_problem['results']['counts'])
