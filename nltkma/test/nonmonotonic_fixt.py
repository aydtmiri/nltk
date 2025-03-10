def setup_module():
    import pytest
    from nltkma.inference.mace import Mace

    try:
        m = Mace()
        m._find_binary("mace4")
    except LookupError as e:
        pytest.skip(
            "Mace4/Prover9 is not available so nonmonotonic.doctest was skipped"
        )
