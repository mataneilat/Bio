
from prody import Gamma


class RaptorGamma(Gamma):

    def __init__(self, raptor_matrix, cutoff, alpha):
        self.raptor_matrix = raptor_matrix
        self.cutoff = cutoff
        self.alpha = alpha

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        raptor_score = self.raptor_matrix[i, j]
        cutoff_squared = self.cutoff * self.cutoff
        return self.alpha * raptor_score + (1 - self.alpha) * (1 - dist2 / cutoff_squared)


class SquaredDistanceGamma(Gamma):

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def getRadii(self):
        pass

    def getGamma(self):
        pass

    def gamma(self, dist2, i, j):
        cutoff_squared = self.cutoff * self.cutoff
        s = 1 - dist2 / cutoff_squared
        return s * s
