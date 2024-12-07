from src.domain.grid.model import Grid


def test_export_under_limit():
    grid = Grid(10.0, 5.0)
    actual = grid.export_power(3.0, 2.0)
    assert actual == 3.0
    assert grid.exported_energy == 6.0
    assert grid.imported_energy == 0.0


def test_export_over_limit():
    grid = Grid(10.0, 5.0)
    actual = grid.export_power(8.0, 2.0)
    assert actual == 5.0
    assert grid.exported_energy == 10.0
    assert grid.imported_energy == 0.0
