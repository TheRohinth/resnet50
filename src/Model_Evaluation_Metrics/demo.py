from tensorflow_addons.metrics import F1Score, CohenKappa

print(
    CohenKappa(
        num_classes = 7,
        name = 'cohen_kappa',
        weightage = None,
        sparse_labels = True,
        regression = False
    )
)

print(
    F1Score(
        num_classes = 7,
        average = None,
        threshold = None,
        name = 'f1_score'
    )
)