from sympy import symbol, Function, Number
from modulus.sym.eq.pde import PDE


class WaveEquation1D(PDE):
    name = "WaveEquation1D"
    
    def __init__(self, c=1.0):
        #c is the speed of the wave in the 1D equation 
        x = symbol("x")
        t = symbol("t")
        
        input_variables = {"x": x, "t": t}
        
        u = Function("u")(*input_variables) ## this equation will help define u in terms of the input variables 
        
        
    