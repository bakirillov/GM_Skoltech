from pandas import DataFrame
from keras.models import Model
import matplotlib.pyplot as plt
import NPEET.entropy_estimators as ee

class IB():
    
    def __init__(self, k=3):
        self.I = ee.mi
        self.k = k
        
    def coords(self, inp, lab, rep):
        """Returns I(inp, lab) and I(inp, rep)"""
        IXT = self.I(rep, inp, k=self.k)
        ITY = self.I(rep, ee.vectorize(lab), k=self.k)
        return([IXT, ITY])
    
    @staticmethod
    def representation(layer):
        """Returns function to access representation from layer"""
        return(
            lambda m: IB.compile_rep(Model(inputs=m.input, outputs=m.get_layer(layer).output))
        )
    
    @staticmethod
    def compile_rep(m):
        m.compile("adam", "categorical_crossentropy")
        return(m)
    
    @staticmethod
    def get_all_names(model):
        nms = [a.name for a in model.layers]
        return(nms[1:])
    
    @staticmethod
    def get_all_reps(model):
        nms = IB.get_all_names(model)
        reps = [IB.representation(a) for a in nms]
        return(reps)
    
    @staticmethod
    def tradeoff(inp_rep, inp_lab, beta):
        """Returns IB tradeoff"""
        return(inp_rep - inp_lab*beta)
    
    @staticmethod
    def to_plane(m):
        return(
            DataFrame(m, columns=["I(X,T)","I(T,Y)"])
        )
    
    @staticmethod
    def plot(plane):
        """Plot infoplane for network"""
        plt.title("Information Plane")
        plt.xlabel("I(Input,Representation)")
        plt.ylabel("I(Representation,Label)")
        plt.grid(True)
        plt.scatter(plane["I(X,T)"], plane["I(T,Y)"], color=plane["color"])
        plt.show()
