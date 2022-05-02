from skfem import *
from skfem.visuals.matplotlib import *
from skfem.helpers import dot, grad
import numpy as np

basis = Basis(MeshTri().refined(3), ElementTriP1())
a = BilinearForm(lambda u, v, _: dot(grad(u), grad(v)))
L = LinearForm(lambda v, w: (4 - 2 * (w.x[0] ** 2 + w.x[1] ** 2)) * v)
y = solve(*condense(a.assemble(basis), L.assemble(basis), D=basis.get_dofs()))
ax = draw(basis)
plot(basis, y, ax=ax, nrefs=2, colorbar=True, shading="gouraud")
plt.show()
plt.savefig("fem.pdf")
