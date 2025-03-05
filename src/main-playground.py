## VSCode: shift enter to run a line in vscode interactive window mode.
## Zed: repl: run, then ctrl+shift+enter to execute each cell

# %%
import torch

# %% Create a random tensor
x = torch.rand((2,3))
x


# %% Permutation of Tensors, e.g. transpose
torch.einsum("ij->ji", x)

# %% Summation of Tensors
torch.einsum("ij->", x)

# %% Column sum
torch.einsum("ij->j", x)

# %% Row sum
torch.einsum("ij->i", x)

# %% Matrix-Vector Multiplication
v = torch.rand((1,3))
torch.einsum("ij,kj->ik", x, v)

# %% Matrix-Matrix Multiplication
torch.einsum("ij,kj->ik", x, x)  # %% 2x2: 2x3 X 3x2

# %% Dot product first row with first row of matrix
torch.einsum("i,i->", x[0], x[0])

# %% Dot product with matrix
torch.einsum("ij,ij->", x, x)

# %% Hadamard Product (element-wise multiplication)
torch.einsum("ij,ij->ij", x, x)

# %% Outer product
a = torch.rand((3))
b = torch.rand((5))
a
b
torch.einsum("i,j->ij", a,b)


# %% Batch matrix Multiplication
a = torch.rand((3,2,5))
b = torch.rand((3,5,3))
a
b

torch.einsum("ijk,ikl->ijl", a, b)

# %% Matrix Diagonal
x = torch.rand((3,3))
x
torch.einsum("ii->i", x)


# %% Matrix Trace
torch.einsum("ii->", x)

# %%
def main():
    print("Hello from einsum-playground!")


if __name__ == "__main__":
    main()
