def test_label_length():
    assert (
        len(labels_class)
        == len(preds_class_probability)
        == len(preds_class)
        == len(labels_family)
    ), "Error: Lengths of labels, preds_class_probability, preds_class, and labels_family are not the same."
