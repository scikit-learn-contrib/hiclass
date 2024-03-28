"""
=========================================
Explanations Filtering
=========================================

An example showing how to filter data returned by Explainer
"""

''''
# Define a level
level = 1

# Use .sel() method
level_1_explanations = explanations.sel(level=1)

# print(level_1_explanations)

# print(level_1_explanations['classes'].values)

level_1_data = explanations.sel(level=1)
# Извлекаем классы, соответствующие первому уровню иерархии
# Выбираем данные для первого уровня иерархии

# print("PSDA\n\n\n\n")
# print(explanations.sel(level=1)['shap_values'].values)

classes_level_1 = level_1_data["classes"]
print(classes_level_1.values)

shp = level_1_data.sel(class_="Cold")["shap_values"].values
print(shp[:, 4])
# for cl in classes_level_1:
#    print(cl, level_1_explanations.sel(class_=cl)['shap_values'].values)
# desired_class = 'Cold_Allergy'
# print(type(classes_level_1))

# for a in classes_level_1:
#    print(a.values)
'''