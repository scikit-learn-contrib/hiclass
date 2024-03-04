import numpy as np
from hiclass.probability_combiner.ProbabilityCombiner import ProbabilityCombiner

class ArithmeticMeanCombiner(ProbabilityCombiner):
    def combine(self, proba):
        res = [proba[0]]
        sums = [proba[0]]
        for level in range(1, self.classifier.max_levels_):
            print(level)
            level_probs = np.zeros_like(proba[level])
            level_sum = np.zeros_like(proba[level])
            for node in self.classifier.classes_[level]:
                predecessor = list(self.classifier.hierarchy_.predecessors(node))[0]
                predecessor_index = self.classifier.class_to_index_mapping_[level-1][predecessor]
                index = self.classifier.class_to_index_mapping_[level][node]

                level_sum[:, index] += proba[level][:, index] + sums[level-1][:, predecessor_index]
                level_probs[:, index] = level_sum[:, index] / (level+1)

            res.append(level_probs)
            sums.append(level_sum)
        return res
