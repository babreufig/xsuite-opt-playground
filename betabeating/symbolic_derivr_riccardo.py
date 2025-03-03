import xtrack as xt

lhc = xt.Multiline.from_json("lhc.json")

class FakeQuad:
    pass

vv = "dqy.b1_op"

k1 = []
myelems = {}
for dd in lhc.ref_manager.find_deps([lhc.vars[vv]]):
    if dd.__class__.__name__ == "AttrRef" and dd._key == "k1":
        k1.append((dd._owner._key, dd._expr))
        myelems[dd._owner._key] = FakeQuad()

fdef = lhc.ref_manager.mk_fun("myfun", a=lhc.vars[vv])
gbl = {
    "vars": lhc.ref_manager.containers["vars"]._owner.copy(),
    "element_refs": myelems,
}
lcl = {}
exec(fdef, gbl, lcl)
fff = lcl["myfun"]

import sympy

a = sympy.var("a")
fff(a)
for kk, expr in k1:
    print(kk, "k1", expr, gbl["element_refs"][kk].k1.diff(a))