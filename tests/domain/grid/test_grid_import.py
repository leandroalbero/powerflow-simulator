from src.domain.grid.model import Grid


def test_import_under_limit():
    grid = Grid(10.0, 5.0)
    actual = grid.import_power(8.0, 2.0)
    assert actual == 8.0
    assert grid.imported_energy == 16.0
    assert grid.exported_energy == 0.0


def test_import_over_limit():
    grid = Grid(10.0, 5.0)
    actual = grid.import_power(15.0, 2.0)
    assert actual == 10.0
    assert grid.imported_energy == 20.0
    assert grid.exported_energy == 0.0
