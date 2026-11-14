"""Expose the src package to the existing standalone cyclone commands."""

from pathlib import Path

_source = Path(__file__).resolve().parents[1] / "src" / "advection_ft"
__path__.insert(0, str(_source))
_init = _source / "__init__.py"
exec(compile(_init.read_bytes(), str(_init), "exec"))
