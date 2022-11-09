.. _metrics-overview:

Metrics
====================

According to [1]_, the use of flat classification metrics might not be adequate to give enough insight of which algorithm is better at classifying hierarchical data. Hence, in HiClass we implemented the metrics of hierarchical precision (hP), hierarchical recall (hR) and hierarchical F-score (hF), which are extensions of the renowned metrics of precision, recall and F-score, but tailored to the hierarchical classification scenario. These hierarchical counterparts were initially proposed by [2]_, and are defined as follows:

:math:`\displaystyle{hP = \frac{\sum_i|\alpha_i\cap\beta_i|}{\sum_i|\alpha_i|}}`, :math:`\displaystyle{hR = \frac{\sum_i|\alpha_i\cap\beta_i|}{\sum_i|\beta_i|}}`, :math:`\displaystyle{hF = \frac{2 \times hP \times hR}{hP + hR}}`

where :math:`\alpha_i` is the set consisting of the most specific classes predicted for test example :math:`i` and all their ancestor classes, while :math:`\beta_i` is the set containing the true most specific classes of test example :math:`i` and all their ancestors, with summations computed over all test examples.

.. [1] Silla, C. N., & Freitas, A. A. (2011). A survey of hierarchical classification across different application domains. Data Mining and Knowledge Discovery, 22(1), 31-72.

.. [2] Kiritchenko, S., Matwin, S., Nock, R., & Famili, A. F. (2006, June). Learning and evaluation in the presence of class hierarchies: Application to text categorization. In Conference of the Canadian Society for Computational Studies of Intelligence (pp. 395-406). Springer, Berlin, Heidelberg.
