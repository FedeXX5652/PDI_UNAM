import math

class Binomial:
    def __init__(self, N) -> None:
        self.N = N
        self.coefficients = self.calculate_coefficients()
    
    def calculate_coefficients(self) -> list:
        coefficients = []
        for k in range(self.N + 1):
            coeff = math.comb(self.N, k)
            coefficients.append(coeff)
        return coefficients