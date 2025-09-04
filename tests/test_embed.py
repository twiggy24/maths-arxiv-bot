from embedder import embed_texts, EMBED_DIM

def test_embed_shape():
    vecs = embed_texts([
        "Picard group of a smooth projective variety",
        "Néron–Severi group is finitely generated over Z"
    ])
    assert len(vecs) == 2
    assert len(vecs[0]) == EMBED_DIM  # should be 3072
    assert isinstance(vecs[0][0], float)


