from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

from src import FeatureMatcher
from src.utils.loadData import loadFrames


def basicModelMaybeToBeUsed():
    # Create the model with edges specified as tuples (parent, child)
    dentist_model = BayesianModel([('Cavity', 'Toothache'),
                                   ('Cavity', 'Catch')])
    # Create tabular CPDs, values has to be 2-D array
    cpd_cav = TabularCPD('Cavity', 2, [[0.2], [0.8]])
    cpd_too = TabularCPD('Toothache', 2, [[0.6, 0.1],
                                          [0.4, 0.9]],
                         evidence=['Cavity'], evidence_card=[2])
    cpd_cat = TabularCPD('Catch', 2, [[0.9, 0.2],
                                      [0.1, 0.8]],
                         evidence=['Cavity'], evidence_card=[2])
    # Add CPDs to model
    dentist_model.add_cpds(cpd_cav, cpd_too, cpd_cat)

    print('Check model :', dentist_model.check_model())

    print('Independencies:\n', dentist_model.get_independencies())

    # Initialize inference algorithm
    dentist_infer = VariableElimination(dentist_model)

    # Some exemple queries
    q = dentist_infer.query(['Toothache'])
    print('P(Toothache) =\n', q)

    q = dentist_infer.query(['Cavity'])
    print('P(Cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': 0})
    print('P(Toothache | cavity) =\n', q)

    q = dentist_infer.query(['Toothache'], evidence={'Cavity': 1})
    print('P(Toothache | ~cavity) =\n', q)


def main():
    data = loadFrames("./data")
    oldROI = data["c6s1_002901.jpg"].getSecondROI()
    newROI = data["c6s1_002926.jpg"].getSecondROI()
    FM = FeatureMatcher.FeatureMatcher()
    matchesNum = FM.featureMatchVis(oldROI, newROI)


if __name__ == '__main__':
    main()
