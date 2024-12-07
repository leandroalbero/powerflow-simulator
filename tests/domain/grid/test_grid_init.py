from src.domain.grid.model import Grid


def test_grid_initialization():
    grid = Grid(10.0, 5.0)
    assert grid.max_import == 10.0
    assert grid.max_export == 5.0
    assert grid.imported_energy == 0.0
    assert grid.exported_energy == 0.0


def test_mixed_operations():
    grid = Grid(10.0, 5.0)
    grid.import_power(8.0, 1.0)
    grid.export_power(3.0, 1.0)
    assert grid.imported_energy == 8.0
    assert grid.exported_energy == 3.0
