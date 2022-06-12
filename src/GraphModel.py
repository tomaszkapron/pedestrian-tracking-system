from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination


def createModel():
    # Create the model with edges specified as tuples (parent, child)
    matching_model = BayesianModel([('ObjMatch', 'FeatMatch'),
                                    ('ObjMatch', 'dir'),
                                    ('ObjMatch', 'hist')])
    # Create tabular CPDs, values has to be 2-D array
    cpd_obj = TabularCPD('ObjMatch', 2, [[0.4], [0.6]])
    cpd_feat = TabularCPD('FeatMatch', 4, [[0.358, 0.009],
                                           [0.32, 0.043],
                                           [0.19, 0.44],
                                           [0.132, 0.508]],
                          evidence=['ObjMatch'], evidence_card=[2])
    cpd_dir = TabularCPD('dir', 2, [[0.6, 0.2],
                                    [0.4, 0.8]],
                         evidence=['ObjMatch'], evidence_card=[2])
    cpd_hist = TabularCPD('hist', 6, [[0.33, 0.004],
                                      [0.29, 0.008],
                                      [0.22, 0.04],
                                      [0.14, 0.2],
                                      [0.017, 0.36],
                                      [0.003, 0.388]],
                          evidence=['ObjMatch'], evidence_card=[2])
    # Add CPDs to model
    matching_model.add_cpds(cpd_obj, cpd_feat, cpd_dir, cpd_hist)

    # Initialize inference algorithm
    return VariableElimination(matching_model)


class GraphModel:
    def __init__(self):
        self.model = createModel()

    def query(self, featEvidence, histEvidence, dirEvidence=None):
        if dirEvidence:
            q = self.model.query(['ObjMatch'], evidence={'dir': dirEvidence,
                                                         'FeatMatch': featEvidence,
                                                         'hist': histEvidence})
            return q

        q = self.model.query(['ObjMatch'], evidence={'FeatMatch': featEvidence, 'hist': histEvidence})
        return q.values[1]
